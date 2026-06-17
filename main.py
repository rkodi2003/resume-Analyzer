from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import json
import os
import pypdf
import io

app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def run_analysis(resume: str, job_description: str) -> dict:
    if not resume.strip() or not job_description.strip():
        raise HTTPException(status_code=400, detail="Resume and job description are required")

    prompt = f"""You are an expert HR analyst and career coach. Analyze the resume against the job description below.

Return ONLY a valid JSON object with exactly this structure:
{{
  "match_score": <integer 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "matched_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "suggestions": [
    {{ "title": "<short title>", "detail": "<actionable advice>" }},
    ...
  ],
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "verdict": "<one of: Strong Match | Good Match | Partial Match | Weak Match>"
}}

RESUME:
{resume}

JOB DESCRIPTION:
{job_description}

Return only the JSON. No explanation, no markdown, no code blocks."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    return run_analysis(payload.resume, payload.job_description)


@app.post("/api/analyze-pdf")
async def analyze_pdf(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
):
    if not resume_pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await resume_pdf.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    return run_analysis(resume_text, job_description)
