exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: "Method not allowed" })
    };
  }
  try {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      throw new Error("AI service is not configured.");
    }
    const { audioBase64, fileName, mimeType } =
      JSON.parse(event.body || "{}");
    if (!audioBase64 || !fileName) {
      return {
        statusCode: 400,
        body: JSON.stringify({
          error: "Please provide an audio file."
        })
      };
    }
    const audioBuffer = Buffer.from(audioBase64, "base64");
    // Initial MVP limit: 8 MB.
    // Long recordings will use chunked cloud processing later.
    if (audioBuffer.length > 8 * 1024 * 1024) {
      return {
        statusCode: 413,
        body: JSON.stringify({
          error: "Audio is too large for this initial test. Use a smaller audio file."
        })
      };
    }
    const form = new FormData();
    form.append(
      "file",
      new Blob([audioBuffer], {
        type: mimeType || "audio/mpeg"
      }),
      fileName
    );
    form.append("model", "whisper-large-v3-turbo");
    form.append("response_format", "verbose_json");
    form.append("timestamp_granularities[]", "segment");
    const response = await fetch(
      "https://api.groq.com/openai/v1/audio/transcriptions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`
        },
        body: form
      }
    );
    const result = await response.json();
    if (!response.ok) {
      return {
        statusCode: response.status,
        body: JSON.stringify({
          error: result.error?.message || "Transcription failed."
        })
      };
    }
    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        success: true,
        transcript: result.text,
        segments: result.segments || [],
        duration: result.duration || null
      })
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({
        error: error.message || "Something went wrong."
      })
    };
  }
};
