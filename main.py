from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import json
import os

app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    if not payload.resume.strip() or not payload.job_description.strip():
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
{payload.resume}

JOB DESCRIPTION:
{payload.job_description}

Return only the JSON. No explanation, no markdown, no code blocks."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

    return result
