# app.py
import os
import io
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests

# ---------- CONFIG ----------
load_dotenv()  # initial load

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("docsage-backend")

PORT = int(os.getenv("PORT", 80))

OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory store for last document text
DOC_TEXT: str = ""


# ---------- Pydantic Models ----------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# ---------- Helpers ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF uploaded in memory."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("No extractable text in PDF")
        return text
    except Exception as e:
        logger.exception("PDF parsing failed")
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")


def _call_openrouter(messages, max_tokens: int = 600) -> str:
    """
    Low-level helper: send messages to OpenRouter and return text content.
    Also supports demo-mode fallback if key missing.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        # Fallback demo string – the caller can format if needed
        return "[DEMO_MODE_NO_KEY]"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=40
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            logger.error("No choices in OpenRouter response: %s", data)
            return "[Error] Model returned no choices."

        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        return content.strip() or "[Error] Model returned empty content."
    except requests.exceptions.RequestException as e:
        logger.exception("Network error calling OpenRouter")
        raise HTTPException(status_code=502, detail=f"LLM network error: {e}")
    except Exception as e:
        logger.exception("Unexpected error calling OpenRouter")
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")


def call_openrouter_qa(question: str, context: str) -> str:
    """Classic Q&A using document context."""
    result = _call_openrouter(
        [
            {
                "role": "system",
                "content": (
                    "You are DOCsage, an assistant that answers questions ONLY using the "
                    "provided document context. If the answer is not clearly in the "
                    'document, reply with: "I’m not sure based on this document."'
                ),
            },
            {
                "role": "user",
                "content": (
                    "DOCUMENT CONTEXT:\n"
                    f"{context}\n\n"
                    "QUESTION:\n"
                    f"{question}\n\n"
                    "Answer in 3–6 concise sentences."
                ),
            },
        ]
    )

    if result == "[DEMO_MODE_NO_KEY]":
        # Nice demo response if .env key is missing
        return (
            "DOCsage (Demo Mode): OPENROUTER_API_KEY is not set in .env, "
            "so this is a dummy answer.\n\n"
            f"Question: {question}\n\n"
            "Context preview:\n"
            + context[:400]
        )

    return result


def call_openrouter_summary(context: str) -> str:
    """Summarize the document into a few bullet points."""
    result = _call_openrouter(
        [
            {
                "role": "system",
                "content": (
                    "You are DOCsage, a document summarizer. You create short, clear "
                    "summaries using bullet points."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following document in 5–7 short bullet points.\n\n"
                    "DOCUMENT:\n"
                    f"{context}"
                ),
            },
        ]
    )

    if result == "[DEMO_MODE_NO_KEY]":
        return "Summary demo: add OPENROUTER_API_KEY in .env to get real summaries."

    return result


def call_openrouter_keywords(context: str) -> str:
    """Extract important keywords / key phrases from document."""
    result = _call_openrouter(
        [
            {
                "role": "system",
                "content": (
                    "You are DOCsage, a keyword extractor. You identify key topics "
                    "and return them as a comma-separated list."
                ),
            },
            {
                "role": "user",
                "content": (
                    "From the following document, extract 10–20 important keywords or "
                    "key phrases (skills, technologies, topics, entities). "
                    "Return ONLY a comma-separated list.\n\n"
                    "DOCUMENT:\n"
                    f"{context}"
                ),
            },
        ]
    )

    if result == "[DEMO_MODE_NO_KEY]":
        return "keywords_demo, add_key_in_env"

    return result


def call_openrouter_suggestions(context: str) -> str:
    """Give smart suggestions based on document type (resume/notes/report/etc)."""
    result = _call_openrouter(
        [
            {
                "role": "system",
                "content": (
                    "You are DOCsage, a document coach. You analyze a document and "
                    "give practical, actionable suggestions."
                ),
            },
            {
                "role": "user",
                "content": (
                    "First, in one line, guess what type of document this is "
                    "(e.g., resume, academic notes, research paper, business report, other).\n"
                    "Then give 5–8 clear suggestions on how to improve this document "
                    "for its purpose.\n\n"
                    "DOCUMENT:\n"
                    f"{context}"
                ),
            },
        ]
    )

    if result == "[DEMO_MODE_NO_KEY]":
        return (
            "Suggestions demo: add OPENROUTER_API_KEY in .env to get real suggestions."
        )

    return result


# ---------- FastAPI App ----------
app = FastAPI(title="DOCsage - AI PDF Q&A + Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the single-page HTML (with embedded CSS + JS)."""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and store extracted text in memory."""
    global DOC_TEXT

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    DOC_TEXT = extract_text_from_pdf(file_bytes)

    preview = DOC_TEXT[:1000]
    return {
        "message": "File uploaded and text extracted successfully",
        "textPreview": preview,
        "totalLength": len(DOC_TEXT),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask_docsage(payload: AskRequest):
    """Answer a question based on the last uploaded document."""
    global DOC_TEXT

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    if not DOC_TEXT:
        raise HTTPException(
            status_code=400, detail="No document uploaded yet. Please upload a PDF."
        )

    context = DOC_TEXT[:MAX_CONTEXT_CHARS]
    answer = call_openrouter_qa(question, context)
    return AskResponse(answer=answer)


@app.get("/api/summary")
def get_summary():
    """Return an automatic summary of the current document."""
    global DOC_TEXT
    if not DOC_TEXT:
        raise HTTPException(
            status_code=400, detail="No document uploaded yet. Please upload a PDF."
        )
    context = DOC_TEXT[:MAX_CONTEXT_CHARS]
    summary = call_openrouter_summary(context)
    return {"summary": summary}


@app.get("/api/keywords")
def get_keywords():
    """Return extracted keywords."""
    global DOC_TEXT
    if not DOC_TEXT:
        raise HTTPException(
            status_code=400, detail="No document uploaded yet. Please upload a PDF."
        )
    context = DOC_TEXT[:MAX_CONTEXT_CHARS]
    raw = call_openrouter_keywords(context)

    # Try to split into a list
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    return {"raw": raw, "keywords": parts}


@app.get("/api/suggest")
def get_suggestions():
    """Return smart improvement suggestions based on document type."""
    global DOC_TEXT
    if not DOC_TEXT:
        raise HTTPException(
            status_code=400, detail="No document uploaded yet. Please upload a PDF."
        )
    context = DOC_TEXT[:MAX_CONTEXT_CHARS]
    suggestions = call_openrouter_suggestions(context)
    return {"suggestions": suggestions}


@app.get("/health")
def health():
    """Simple health check."""
    load_dotenv()
    key_set = bool(os.getenv("OPENROUTER_API_KEY"))
    return {
        "ok": True,
        "openrouter_key_set": key_set,
        "model": OPENROUTER_MODEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
