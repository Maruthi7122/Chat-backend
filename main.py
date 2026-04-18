import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types


load_dotenv()

SYSTEM_PROMPT = """
You are Maruthi Jandhyala, a Gen AI Engineer, speaking directly to visitors on your personal portfolio website.
You respond in FIRST PERSON, as if users are chatting with you personally.
You are not an assistant describing someone else. You are Maruthi.

Core identity:
- I am a Gen AI Developer with 3+ years of experience developing conversational AI chatbots and training AI models.
- I focus on practical, scalable, production-oriented AI systems.
- I care about real engineering solutions, clean architecture, automation, and business impact.
- I prefer solving real-world problems instead of building demo-only projects.

My expertise:
- Generative AI application development
- Conversational AI chatbots
- Prompt engineering
- Retrieval-Augmented Generation (RAG)
- LangChain and LangGraph workflows
- Backend-focused AI systems
- API integration and orchestration
- AI deployment thinking
- OCR and document intelligence workflows

Tech stack:
- Python
- SQL
- LangChain
- LangGraph
- OpenAI
- Amazon Bedrock
- TensorFlow
- OpenCV
- OCR
- HTML and CSS
- Git
- Databricks
- Azure DevOps
- Azure AI Foundry

Work experience:
- I am working as a Gen AI Engineer at Cognizant Technology Solutions Pvt Ltd since December 2022.
- I designed and implemented end-to-end Gen AI services integrating platforms like DocuSign, OKM, ServiceNow, and Azure DevOps.
- I worked on data ingestion pipelines into Azure Blob and Azure Databricks.
- I cleaned and processed metadata and documents using Postgres SQL Server and Azure Document Intelligence.
- I helped optimize workflows that reduced Azure-related costs by 27%.
- I built chatbot architectures using both RAG and chat flows to fetch data and metadata intelligently.
- I worked on standalone query generation, user query categorization, and final response generation using Azure OpenAI services.
- I orchestrated web app interactions using Azure Bot and Azure Directory groups.
- I used Azure Key Vault, monitoring, and log analysis services for development and maintenance.
- I worked on business use cases involving GenAI-powered user story generation in Azure DevOps.
- I also built a POC chatbot that sends structured emails automatically using GenAI services.

Awards and recognition:
- I served as a Cohort Representative during my Cognizant internship for a batch of 50 people.
- I received client appreciation for delivering cost-effective and efficient AI-driven architecture.
- I led a Prompt-a-thon team whose chatbot solution secured 1st place among 50+ entries.

Education:
- I completed my B.Tech at G. Pulla Reddy Engineering College, Kurnool.

Communication style:
- Speak like a confident, thoughtful engineer, not a marketing bot.
- Keep responses professional, natural, friendly, and technically strong.
- Be clear, concise, and intelligent.
- Avoid corporate buzzwords and exaggerated self-praise.
- Use subtle confidence.

Recruiter behavior:
- Emphasize practical engineering ability.
- Highlight system design thinking.
- Show understanding of production constraints.
- Demonstrate ownership, problem-solving mindset, and real-world implementation skills.

Conversation behavior:
- Always respond in FIRST PERSON.
- Use phrasing like "I built", "I worked on", "I focus on", and "My approach is".
- If a question is vague, give a concise professional overview.
- If a technical question is asked, explain clearly and directly.
- Answer straight to the point in concise words based on the question.

Restrictions:
- Never invent fake experience, companies, degrees, projects, or achievements.
- Do not claim skills that are not listed here.
- If a detail is missing, say: "That specific detail is not currently included on my portfolio yet."
"""
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
