"""
Career Advisor AI - FastAPI backend entry point.

Stage 1 (this file, current scope): Resume upload -> parsed structured JSON.
Later stages will add: JD parsing endpoint, fit-scoring endpoint,
gap-analysis endpoint, and a final /analyze endpoint that runs the
whole LangGraph pipeline in one call.
"""

import os
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment variables — must happen before agents import Groq client

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.text_extraction import extract_resume_text
from agents.resume_parser import parse_resume
from agents.jd_parser import parse_job_description
from agents.fit_scorer import score_fit
from agents.gap_analyzer import analyze_gap
from agents.resume_improver import improve_resume
from agents.orchestrator import run_full_analysis
from schemas.resume_schema import ParsedResume
from schemas.jd_schema import ParsedJobDescription
from schemas.fit_schema import FitScoreResult
from schemas.report_schema import CareerAdvisorReport

app = FastAPI(title="Career Advisor AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying, same as you did for MedScribe
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "career-advisor-ai", "stage": "resume-parser"}


@app.post("/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """
    Accepts a .pdf or .txt resume file, returns structured parsed JSON.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are supported.")

    # Save upload to a temp file so PyMuPDF can read it by path
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        raw_text = extract_resume_text(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

        parsed = parse_resume(raw_text)
        return parsed.model_dump()

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")
    finally:
        os.remove(tmp_path)


class JobDescriptionInput(BaseModel):
    text: str


@app.post("/parse-jd")
async def parse_jd_endpoint(payload: JobDescriptionInput):
    """
    Accepts raw job description text (pasted, not a file -- JDs are almost
    always copy-pasted from a job board) and returns structured parsed JSON.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Job description text cannot be empty.")

    try:
        parsed = parse_job_description(payload.text)
        return parsed.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"JD parsing failed: {e}")


class FitScoreInput(BaseModel):
    resume: ParsedResume
    job_description: ParsedJobDescription


@app.post("/score-fit")
async def score_fit_endpoint(payload: FitScoreInput):
    """
    Accepts an already-parsed resume and job description (from /parse-resume
    and /parse-jd) and returns a numeric fit score with matched/missing skill
    breakdown. Note: first call after server start downloads the embedding
    model (~80MB) if not cached yet -- expect a delay on the very first request.
    """
    try:
        result = score_fit(payload.resume, payload.job_description)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fit scoring failed: {e}")


class GapAnalysisInput(BaseModel):
    resume: ParsedResume
    job_description: ParsedJobDescription
    fit_result: FitScoreResult


@app.post("/analyze-gap")
async def analyze_gap_endpoint(payload: GapAnalysisInput):
    """
    Accepts the parsed resume, parsed JD, and the fit score result (from
    /parse-resume, /parse-jd, /score-fit) and returns plain-language advice:
    strengths, gaps, actionable suggestions, and resume wording tips.
    """
    try:
        result = analyze_gap(payload.resume, payload.job_description, payload.fit_result)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {e}")


@app.post("/improve-resume")
async def improve_resume_endpoint(payload: GapAnalysisInput):
    """
    Accepts the parsed resume, parsed JD, and fit score (same input shape as
    /analyze-gap) and returns concrete rewritten bullet suggestions the
    candidate can paste directly into their resume.
    """
    try:
        result = improve_resume(payload.resume, payload.job_description, payload.fit_result)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Resume improvement failed: {e}")


@app.post("/analyze", response_model=CareerAdvisorReport)
async def analyze_endpoint(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
):
    """
    The single-call version: upload a resume file (.pdf or .txt) + paste JD
    text as a form field, and get back the full report (parsed resume,
    parsed JD, fit score, gap analysis, and resume improvement suggestions)
    in one request.

    Note: this runs all 5 agents (2 parser Groq calls, embeddings, then gap
    analysis + resume improvement running in parallel) -- expect this to
    take longer than any single individual endpoint (typically 10-30 seconds).
    """
    if not resume_file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Resume must be a .pdf or .txt file.")
    if not jd_text.strip():
        raise HTTPException(status_code=422, detail="Job description text cannot be empty.")

    suffix = os.path.splitext(resume_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(resume_file.file, tmp)
        tmp_path = tmp.name

    try:
        resume_text = extract_resume_text(tmp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text from the resume file.")

        final_state = run_full_analysis(resume_text, jd_text)

        return CareerAdvisorReport(
            parsed_resume=final_state["parsed_resume"],
            parsed_job_description=final_state["parsed_jd"],
            fit_result=final_state["fit_result"],
            gap_analysis=final_state["gap_analysis"],
            resume_improvements=final_state["resume_improvements"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full analysis pipeline failed: {e}")
    finally:
        os.remove(tmp_path)
