"""
api/main.py
────────────
FastAPI backend for the HR Shortlisting Agent.

Endpoints:
  POST /analyze         – JD + resumes → ranked candidates
  POST /override        – Manual score override
  GET  /results/{id}    – Retrieve a session's results
  GET  /health          – Health check

Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Path setup so agents/core/utils are importable ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.jd_parser import JDParserAgent
from agents.profile_parser import ProfileParserAgent
from agents.ranking_agent import RankingAgent
from agents.scoring_agent import ScoringAgent
from report.generator import ReportGenerator
from utils.linkedin_parser import parse_linkedin
from utils.resume_parser import extract_text

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="HR Resume & LinkedIn Shortlisting Agent",
    description="AI-powered candidate evaluation pipeline with explainable scoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────
# In production, replace with Redis / database
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# ── Agent singletons ─────────────────────────
_EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
_USE_LLM = os.getenv("USE_LLM", "true").lower() != "false"

jd_agent = JDParserAgent(use_llm=_USE_LLM)
profile_agent = ProfileParserAgent(use_llm=_USE_LLM)
scoring_agent = ScoringAgent(embedding_backend=_EMBEDDING_BACKEND, use_llm=_USE_LLM)
ranking_agent = RankingAgent()
report_gen = ReportGenerator()


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class OverrideRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /analyze")
    candidate_id: str = Field(..., description="candidate_id field from result")
    category: str = Field(
        ...,
        description="One of: skills_match, experience, education, projects, communication"
    )
    new_score: int = Field(..., ge=0, le=10, description="New score 0–10")
    reason: str = Field(..., description="Justification for the override")
    overrider: str = Field("HR Manager", description="Name/role of person overriding")


class AnalysisResponse(BaseModel):
    session_id: str
    job_title: str
    candidates: List[Dict[str, Any]]
    statistics: Dict[str, Any]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "embedding_backend": _EMBEDDING_BACKEND,
        "llm_enabled": _USE_LLM,
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(
    jd_text: str = Form(..., description="Raw job description text"),
    linkedin_text: Optional[str] = Form(
        None, description="LinkedIn profile text or JSON (optional)"
    ),
    resumes: List[UploadFile] = File(
        default=[], description="PDF/DOCX resume files"
    ),
):
    """
    Full pipeline:
      1. Parse JD
      2. Parse all resumes + optional LinkedIn
      3. Score each candidate
      4. Rank candidates
      5. Return ranked list + session ID
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="JD text cannot be empty.")
    if not resumes and not linkedin_text:
        raise HTTPException(
            status_code=400,
            detail="At least one resume or LinkedIn profile must be provided."
        )

    # ── 1. Parse JD ──────────────────────────
    parsed_jd = jd_agent.parse(jd_text)

    # ── 2. Parse candidates ──────────────────
    candidates: List[Dict[str, Any]] = []

    # Resume files
    for upload in resumes:
        raw_bytes = await upload.read()
        try:
            text = extract_text(raw_bytes, filename=upload.filename)
            profile = profile_agent.parse_resume(text, filename=upload.filename)
            candidates.append(profile)
        except Exception as e:
            # Log but don't crash the whole pipeline
            candidates.append({
                "candidate_id": str(uuid.uuid4())[:8],
                "name": upload.filename,
                "source": upload.filename,
                "raw_text": "",
                "skills": [],
                "experience": [],
                "education": [],
                "certifications": [],
                "projects": [],
                "total_experience_years": 0,
                "highest_education_level": "unknown",
                "summary": "",
                "_error": str(e),
            })

    # LinkedIn profile
    if linkedin_text and linkedin_text.strip():
        try:
            li_data = parse_linkedin(linkedin_text)
            profile = profile_agent.parse_linkedin(li_data)
            candidates.append(profile)
        except Exception as e:
            pass  # Skip on error

    # ── 3. Score each candidate ───────────────
    scored = [scoring_agent.score(c, parsed_jd) for c in candidates]

    # ── 4. Rank ───────────────────────────────
    ranked = ranking_agent.rank(scored)
    stats = ranking_agent.summary_stats(ranked)

    # ── 5. Store session ──────────────────────
    session_id = str(uuid.uuid4())
    _SESSION_STORE[session_id] = {
        "jd": parsed_jd,
        "ranked": ranked,
        "stats": stats,
    }

    return AnalysisResponse(
        session_id=session_id,
        job_title=parsed_jd.get("title", "Unknown"),
        candidates=ranked,
        statistics=stats,
    )


@app.post("/override", tags=["Human-in-the-Loop"])
async def override_score(req: OverrideRequest):
    """
    Apply a manual score override to a specific candidate's category score.
    """
    session = _SESSION_STORE.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Find candidate in ranked list
    target = next(
        (c for c in session["ranked"] if c["candidate_id"] == req.candidate_id),
        None
    )
    if not target:
        raise HTTPException(status_code=404, detail="Candidate not found in session.")

    try:
        updated = scoring_agent.apply_override(
            result=target,
            category=req.category,
            new_score=req.new_score,
            reason=req.reason,
            overrider=req.overrider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Re-rank after override
    session["ranked"] = ranking_agent.rank(session["ranked"])
    session["stats"] = ranking_agent.summary_stats(session["ranked"])

    return {
        "message": "Override applied successfully.",
        "updated_candidate": updated,
        "new_rank": updated.get("rank"),
        "new_total_score": updated["total_score"],
        "override_log": updated["override_log"],
    }


@app.get("/results/{session_id}", tags=["Reports"])
async def get_results(session_id: str, format: str = "json"):
    """
    Retrieve results for a session.

    Query params:
      format : "json" (default) | "html"
    """
    session = _SESSION_STORE.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if format == "html":
        html = report_gen.generate_html(
            ranked_candidates=session["ranked"],
            jd=session["jd"],
            stats=session["stats"],
        )
        return HTMLResponse(content=html)

    return JSONResponse(content={
        "session_id": session_id,
        "job_title": session["jd"].get("title", ""),
        "statistics": session["stats"],
        "candidates": session["ranked"],
    })


@app.get("/sessions", tags=["System"])
async def list_sessions():
    """List all active session IDs."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "job_title": data["jd"].get("title", ""),
                "candidate_count": len(data["ranked"]),
            }
            for sid, data in _SESSION_STORE.items()
        ]
    }
