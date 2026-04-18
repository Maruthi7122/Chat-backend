import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types


load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Give clear, concise, and accurate answers. "
    "If the request is ambiguous, ask for clarification."
)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    app.config["GENAI_CLIENT"] = genai.Client(api_key=api_key) if api_key else None

    @app.get("/")
    def index():
        return jsonify(
            {
                "message": "Chat backend is running.",
                "endpoints": {
                    "health": "/health",
                    "chat": "/api/ask",
                },
            }
        )

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": DEFAULT_MODEL,
                "gemini_configured": bool(app.config["GENAI_CLIENT"]),
            }
        )

    @app.post("/api/ask")
    def portfolio_chat():
        data = request.get_json(silent=True) or {}
        user_prompt = (data.get("prompt") or "").strip()

        if not user_prompt:
            return jsonify({"response": "Please enter a valid prompt."}), 400

        client = app.config["GENAI_CLIENT"]
        if client is None:
            return (
                jsonify(
                    {
                        "response": "Server is missing GEMINI_API_KEY. Add it to your environment or .env file.",
                    }
                ),
                500,
            )

        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                ),
                contents=user_prompt,
            )
            answer = (response.text or "").strip()
            if not answer:
                answer = "The model returned an empty response."
            return jsonify({"response": answer})
        except Exception as exc:
            return jsonify({"response": f"Gemini request failed: {exc}"}), 500

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
