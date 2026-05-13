"""
agents/scoring_agent.py
────────────────────────
Scoring Agent

Evaluates a candidate profile against a parsed JD using the mandatory rubric:

  Category               Weight
  ─────────────────────────────
  Skills Match            30%
  Experience Relevance    25%
  Education               15%
  Projects / Portfolio    20%
  Communication Quality   10%

Each category produces a score [0–10] + justification.
Final weighted total is in [0–10].

Also supports manual score override with audit logging.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.embedding import EmbeddingEngine
from core.similarity import (
    cosine_similarity,
    hybrid_similarity,
    similarity_to_rubric,
    skill_overlap_score,
)


# ──────────────────────────────────────────────
# Rubric weights (must sum to 1.0)
# ──────────────────────────────────────────────

RUBRIC_WEIGHTS = {
    "skills_match": 0.30,
    "experience": 0.25,
    "education": 0.15,
    "projects": 0.20,
    "communication": 0.10,
}

RECOMMENDATION_THRESHOLDS = {
    "Strong Hire": 8.0,
    "Hire": 6.5,
    "Maybe": 5.0,
    "No Hire": 0.0,
}

EDUCATION_RANK = {
    "phd": 5, "doctorate": 5,
    "master": 4, "msc": 4, "mtech": 4, "mba": 4,
    "bachelor": 3, "bsc": 3, "btech": 3, "be": 3,
    "associate": 2, "diploma": 2,
    "high school": 1, "high_school": 1,
    "any": 0, "unknown": 0,
}


# ──────────────────────────────────────────────
# Scoring Agent
# ──────────────────────────────────────────────

class ScoringAgent:
    """
    Computes a structured rubric score for a candidate vs. a JD.

    Args:
        embedding_backend : "sentence_transformers" | "openai" | "gemini"
        use_llm           : use LLM for semantic justifications
    """

    def __init__(
        self,
        embedding_backend: str = "sentence_transformers",
        use_llm: bool = True,
    ):
        self.engine = EmbeddingEngine(backend=embedding_backend)
        self.use_llm = use_llm

    # ─────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────

    def score(
        self,
        candidate: Dict[str, Any],
        jd: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Score a single candidate against the parsed JD.

        Returns:
            {
              "name": str,
              "candidate_id": str,
              "scores": { category: {score, weight, justification} },
              "total_score": float,
              "recommendation": str,
              "skill_gaps": [...],
              "confidence": float,
              "override_log": []
            }
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # ── 1. Skills Match (30%) ──────────────
        sm_score, sm_just, skill_gaps = self._score_skills(candidate, jd)
        scores["skills_match"] = {
            "score": sm_score,
            "weight": RUBRIC_WEIGHTS["skills_match"],
            "justification": sm_just,
        }

        # ── 2. Experience Relevance (25%) ──────
        exp_score, exp_just = self._score_experience(candidate, jd)
        scores["experience"] = {
            "score": exp_score,
            "weight": RUBRIC_WEIGHTS["experience"],
            "justification": exp_just,
        }

        # ── 3. Education (15%) ─────────────────
        edu_score, edu_just = self._score_education(candidate, jd)
        scores["education"] = {
            "score": edu_score,
            "weight": RUBRIC_WEIGHTS["education"],
            "justification": edu_just,
        }

        # ── 4. Projects / Portfolio (20%) ──────
        proj_score, proj_just = self._score_projects(candidate, jd)
        scores["projects"] = {
            "score": proj_score,
            "weight": RUBRIC_WEIGHTS["projects"],
            "justification": proj_just,
        }

        # ── 5. Communication Quality (10%) ─────
        comm_score, comm_just = self._score_communication(candidate)
        scores["communication"] = {
            "score": comm_score,
            "weight": RUBRIC_WEIGHTS["communication"],
            "justification": comm_just,
        }

        # ── Weighted total ─────────────────────
        total = round(
            sum(v["score"] * v["weight"] for v in scores.values()), 2
        )
        recommendation = self._recommend(total)
        confidence = self._compute_confidence(candidate)

        return {
            "name": candidate.get("name", "Unknown"),
            "candidate_id": candidate.get("candidate_id", ""),
            "source": candidate.get("source", "resume"),
            "scores": scores,
            "total_score": total,
            "recommendation": recommendation,
            "skill_gaps": skill_gaps,
            "confidence": confidence,
            "override_log": [],
        }

    # ─────────────────────────────────────────
    # Individual scorers
    # ─────────────────────────────────────────

    def _score_skills(
        self, candidate: Dict, jd: Dict
    ) -> Tuple[int, str, List[str]]:
        jd_skills = jd.get("required_skills", []) + jd.get("preferred_skills", [])
        cand_skills = candidate.get("skills", [])

        if not jd_skills:
            # No skills in JD → use semantic similarity only
            overlap_pct = 0.5
            matched, missing = [], []
        else:
            overlap_pct, matched, missing = skill_overlap_score(jd_skills, cand_skills)

        # Also compute semantic similarity between full texts
        jd_text = jd.get("raw_text", " ".join(jd_skills))
        cand_text = candidate.get("raw_text", " ".join(cand_skills))
        if jd_text and cand_text:
            jd_vec = self.engine.embed(jd_text[:2000])
            cand_vec = self.engine.embed(cand_text[:2000])
            semantic_sim = cosine_similarity(jd_vec, cand_vec)
        else:
            semantic_sim = overlap_pct

        hybrid = hybrid_similarity(semantic_sim, overlap_pct)
        score = similarity_to_rubric(
            hybrid,
            low_threshold=0.30,
            high_threshold=0.85,
            low_band=(0, 3),
            mid_band=(4, 7),
            high_band=(8, 10),
        )
        if matched:
            just = f"Matched {len(matched)}/{len(jd_skills) or 1} required skills: {', '.join(matched[:5])}"
        else:
            just = f"No direct skill matches found. Semantic similarity: {semantic_sim:.0%}"

        return score, just, missing

    def _score_experience(self, candidate: Dict, jd: Dict) -> Tuple[int, str]:
        exp_years = candidate.get("total_experience_years", 0)
        jd_min = jd.get("experience_years", {}).get("min", 0)
        jd_max = jd.get("experience_years", {}).get("max", 100)
        jd_level = jd.get("experience_level", "")
        jd_domain = jd.get("domain", "")

        # Semantic similarity of experience descriptions vs JD responsibilities
        exp_texts = " ".join(
            e.get("description", "") + " " + e.get("title", "")
            for e in candidate.get("experience", [])
        ).strip()
        resp_text = " ".join(jd.get("responsibilities", []))

        if exp_texts and resp_text:
            e_vec = self.engine.embed(exp_texts[:2000])
            r_vec = self.engine.embed(resp_text[:2000])
            semantic_sim = cosine_similarity(e_vec, r_vec)
        else:
            semantic_sim = 0.5

        # Years gap factor
        if exp_years == 0 and jd_min > 0:
            years_factor = 0.0
        elif exp_years >= jd_min:
            years_factor = min(1.0, exp_years / max(jd_max, 1))
        else:
            years_factor = exp_years / max(jd_min, 1)

        combined = hybrid_similarity(semantic_sim, years_factor)
        score = similarity_to_rubric(combined)

        just = (
            f"{exp_years} yrs experience; JD requires {jd_min}–{jd_max} yrs. "
            f"Domain relevance: {semantic_sim:.0%}."
        )
        return score, just

    def _score_education(self, candidate: Dict, jd: Dict) -> Tuple[int, str]:
        cand_edu_level = candidate.get("highest_education_level", "unknown")
        jd_edu_level = jd.get("education_level", "any")
        cand_certifications = candidate.get("certifications", [])

        cand_rank = EDUCATION_RANK.get(cand_edu_level.lower(), 0)
        jd_rank = EDUCATION_RANK.get(jd_edu_level.lower(), 0)

        cert_bonus = min(len(cand_certifications) * 0.5, 2.0)

        if jd_rank == 0:
            base_score = 7  # No specific requirement
        elif cand_rank >= jd_rank + 1:
            base_score = 9  # Exceeds requirement
        elif cand_rank == jd_rank:
            base_score = 6  # Meets requirement
        elif cand_rank == jd_rank - 1:
            base_score = 4  # Slightly below
        else:
            base_score = 2  # Well below

        score = min(10, int(base_score + cert_bonus))
        just = (
            f"Candidate: {cand_edu_level or 'unknown'} "
            f"(rank {cand_rank}); JD requires: {jd_edu_level} (rank {jd_rank}). "
            f"Certifications: {len(cand_certifications)}."
        )
        return score, just

    def _score_projects(self, candidate: Dict, jd: Dict) -> Tuple[int, str]:
        projects = candidate.get("projects", [])
        jd_skills = jd.get("required_skills", [])
        jd_domain = jd.get("domain", "")

        if not projects:
            return 2, "No projects or portfolio found in resume."

        project_text = " ".join(str(p) for p in projects)
        jd_text = jd.get("raw_text", jd_domain + " " + " ".join(jd_skills))

        if project_text and jd_text:
            p_vec = self.engine.embed(project_text[:2000])
            j_vec = self.engine.embed(jd_text[:2000])
            sim = cosine_similarity(p_vec, j_vec)
        else:
            sim = 0.4

        score = similarity_to_rubric(sim)
        just = (
            f"{len(projects)} project(s) found. "
            f"Relevance to JD domain '{jd_domain}': {sim:.0%}."
        )
        return score, just

    def _score_communication(self, candidate: Dict) -> Tuple[int, str]:
        raw_text = candidate.get("raw_text", "")
        if not raw_text:
            return 4, "No text available for communication assessment."

        word_count = len(raw_text.split())
        # Heuristics: structural markers, sentence variety
        has_sections = len(re.findall(
            r"(?i)(experience|education|skills|summary|projects|certifications)",
            raw_text
        ))
        avg_sent_len = word_count / max(raw_text.count(".") + raw_text.count("!"), 1)
        bullet_ratio = raw_text.count("•") / max(word_count, 1)

        score = 4  # baseline average
        if word_count > 200:
            score += 1
        if has_sections >= 4:
            score += 1
        if 10 <= avg_sent_len <= 25:
            score += 1
        if bullet_ratio > 0.01:
            score += 1
        score = min(score, 10)

        level = "Excellent" if score >= 8 else "Average" if score >= 5 else "Poor"
        just = (
            f"{level} communication. {word_count} words, "
            f"{has_sections} structured sections detected."
        )
        return score, just

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _recommend(self, total: float) -> str:
        for label, threshold in sorted(
            RECOMMENDATION_THRESHOLDS.items(), key=lambda x: -x[1]
        ):
            if total >= threshold:
                return label
        return "No Hire"

    def _compute_confidence(self, candidate: Dict) -> float:
        """Estimate how much structured data was available (0–1)."""
        factors = [
            bool(candidate.get("skills")),
            bool(candidate.get("experience")),
            bool(candidate.get("education")),
            bool(candidate.get("raw_text")),
            bool(candidate.get("summary")),
        ]
        return round(sum(factors) / len(factors), 2)

    # ─────────────────────────────────────────
    # Manual override
    # ─────────────────────────────────────────

    def apply_override(
        self,
        result: Dict[str, Any],
        category: str,
        new_score: int,
        reason: str,
        overrider: str = "HR Manager",
    ) -> Dict[str, Any]:
        """
        Apply a manual score override to a scored result.

        Args:
            result   : existing scored result dict (mutated in place)
            category : one of skills_match, experience, education, projects, communication
            new_score: integer 0–10
            reason   : justification for override
            overrider: name/role of person making override

        Returns:
            Updated result dict with override_log entry appended.
        """
        if category not in result["scores"]:
            raise ValueError(f"Unknown category '{category}'. "
                             f"Valid: {list(result['scores'].keys())}")
        if not (0 <= new_score <= 10):
            raise ValueError("Score must be between 0 and 10.")

        old_score = result["scores"][category]["score"]
        result["scores"][category]["score"] = new_score
        result["scores"][category]["justification"] += (
            f" [OVERRIDE by {overrider}: {reason}]"
        )

        # Recompute total
        result["total_score"] = round(
            sum(v["score"] * v["weight"] for v in result["scores"].values()), 2
        )
        result["recommendation"] = self._recommend(result["total_score"])

        # Audit log
        result["override_log"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "old_score": old_score,
            "new_score": new_score,
            "reason": reason,
            "overrider": overrider,
        })
        return result
