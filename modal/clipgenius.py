import os
import uuid
import subprocess
import requests
import modal

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

APP_NAME = "clipgenius-ai"

volume = modal.Volume.from_name(
    "clipgenius-storage",
    create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "requests",
        "python-multipart"
    )
)

app = modal.App(APP_NAME)

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("clipgenius-secrets")],
    timeout=3600,
    cpu=4,
    memory=8192
)
@modal.asgi_app()
def fastapi_app():

    web = FastAPI(title="ClipGenius AI")

    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    BASE = Path("/data")
    BASE.mkdir(exist_ok=True)

    @web.get("/health")
    def health():
        return {
            "status": "online",
            "service": "ClipGenius AI"
        }

    @web.post("/process")
    async def process_video(video: UploadFile = File(...)):

        if not video.filename:
            raise HTTPException(400, "Video filename is missing")

        if not video.content_type or not video.content_type.startswith("video/"):
            raise HTTPException(400, "Please upload a video file")

        job_id = str(uuid.uuid4())
        job_dir = BASE / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        input_path = job_dir / "input.mp4"
        audio_path = job_dir / "audio.wav"

        size = 0
        max_size = 500 * 1024 * 1024

        try:
            with open(input_path, "wb") as output:
                while True:
                    chunk = await video.read(1024 * 1024)

                    if not chunk:
                        break

                    size += len(chunk)

                    if size > max_size:
                        raise HTTPException(
                            413,
                            "Video exceeds the 500 MB upload limit"
                        )

                    output.write(chunk)

            if size == 0:
                raise HTTPException(400, "The uploaded video is empty")

            # Extract audio
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    "-c:a", "pcm_s16le",
                    str(audio_path)
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise HTTPException(400, "Could not extract video audio")

            # Split audio into manageable chunks
            chunk_dir = job_dir / "audio_chunks"
            chunk_dir.mkdir()

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(audio_path),
                    "-f", "segment",
                    "-segment_time", "480",
                    "-reset_timestamps", "1",
                    "-c:a", "libmp3lame",
                    "-b:a", "64k",
                    str(chunk_dir / "chunk_%03d.mp3")
                ],
                check=True,
                capture_output=True
            )

            api_key = os.environ.get("GROQ_API_KEY")

            if not api_key:
                raise HTTPException(500, "Groq API key is missing")

            transcript_segments = []

            audio_chunks = sorted(chunk_dir.glob("chunk_*.mp3"))

            for index, audio_chunk in enumerate(audio_chunks):

                with open(audio_chunk, "rb") as audio_file:

                    response = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={
                            "Authorization": f"Bearer {api_key}"
                        },
                        files={
                            "file": (
                                audio_chunk.name,
                                audio_file,
                                "audio/mpeg"
                            )
                        },
                        data={
                            "model": "whisper-large-v3-turbo",
                            "response_format": "verbose_json"
                        },
                        timeout=600
                    )

                if response.status_code != 200:
                    raise HTTPException(
                        502,
                        "Transcription failed: " + response.text[:300]
                    )

                data = response.json()

                # Each chunk starts approximately 480 seconds
                offset = index * 480

                for segment in data.get("segments", []):
                    transcript_segments.append({
                        "start": float(segment["start"]) + offset,
                        "end": float(segment["end"]) + offset,
                        "text": segment["text"].strip()
                    })

            if not transcript_segments:
                raise HTTPException(422, "No speech was detected")

            # Find promising 30-second moments
            candidates = []

            keywords = [
                "secret", "mistake", "never", "important",
                "amazing", "shocking", "truth", "money",
                "problem", "believe", "because", "finally",
                "actually", "imagine", "incredible"
            ]

            for segment in transcript_segments:

                start = max(0, segment["start"])
                end = start + 30

                if end > segment["end"] + 30:
                    continue

                text = segment["text"].lower()

                score = sum(
                    1 for word in keywords
                    if word in text
                )

                candidates.append({
                    "start": start,
                    "end": end,
                    "score": score,
                    "text": segment["text"]
                })

            candidates.sort(
                key=lambda item: item["score"],
                reverse=True
            )

            selected = []

            for candidate in candidates:
                overlaps = any(
                    abs(candidate["start"] - item["start"]) < 25
                    for item in selected
                )

                if not overlaps:
                    selected.append(candidate)

                if len(selected) == 3:
                    break

            if not selected:
                selected = [{
                    "start": 0,
                    "end": 30,
                    "score": 0,
                    "text": "Video highlight"
                }]

            # Create vertical clips
            clips = []

            for number, moment in enumerate(selected, start=1):

                clip_name = f"clip_{number}.mp4"
                clip_path = job_dir / clip_name

                duration = moment["end"] - moment["start"]

                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(moment["start"]),
                        "-i", str(input_path),
                        "-t", str(duration),
                        "-vf",
                        "scale=1080:1920:force_original_aspect_ratio=increase,"
                        "crop=1080:1920",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "23",
                        "-c:a", "aac",
                        "-movflags", "+faststart",
                        str(clip_path)
                    ],
                    check=True,
                    capture_output=True
                )

                clips.append({
                    "name": clip_name,
                    "url": f"/clips/{job_id}/{clip_name}",
                    "start": moment["start"],
                    "end": moment["end"],
                    "highlight_text": moment["text"],
                    "score": moment["score"]
                })

            volume.commit()

            return {
                "success": True,
                "job_id": job_id,
                "transcript": transcript_segments,
                "clips": clips,
                "message": "Video processing completed"
            }

        except HTTPException:
            raise

        except Exception as error:
            raise HTTPException(
                500,
                f"Processing error: {str(error)[:300]}"
            )

        finally:
            await video.close()

    @web.get("/clips/{job_id}/{filename}")
    def get_clip(job_id: str, filename: str):

        if not filename.endswith(".mp4") or "/" in filename:
            raise HTTPException(400, "Invalid filename")

        clip_path = BASE / job_id / filename

        if not clip_path.exists():
            raise HTTPException(404, "Clip not found")

        return FileResponse(
            clip_path,
            media_type="video/mp4",
            filename=filename
        )

    return web
