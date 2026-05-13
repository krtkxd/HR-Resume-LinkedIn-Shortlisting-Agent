"""
agents/jd_parser.py
────────────────────
JD Parser Agent

Accepts raw Job Description text and returns a structured JSON:
  {
    "title": str,
    "required_skills": [...],
    "preferred_skills": [...],
    "experience_years": {"min": int, "max": int},
    "experience_level": str,       # e.g. "Senior", "Mid", "Junior"
    "education": str,              # e.g. "Bachelor's in CS"
    "education_level": str,        # normalised: "bachelor", "master", "phd"
    "keywords": [...],
    "domain": str,                 # e.g. "Machine Learning", "Web Development"
    "responsibilities": [...],
    "raw_text": str
  }

Uses LLM (if API key configured) with regex/rule fallback for offline mode.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Skill taxonomy (loaded once)
# ──────────────────────────────────────────────

_SKILL_TAXONOMY: Dict[str, List[str]] = {}
_EDUCATION_LEVELS: Dict[str, int] = {}


def _load_taxonomy() -> None:
    global _SKILL_TAXONOMY, _EDUCATION_LEVELS
    if _SKILL_TAXONOMY:
        return
    skills_path = os.path.join(os.path.dirname(__file__), "..", "data", "skills.json")
    try:
        with open(skills_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _SKILL_TAXONOMY = {k: v for k, v in data.items() if isinstance(v, list)}
        _EDUCATION_LEVELS = data.get("education_levels", {})
    except FileNotFoundError:
        pass  # Taxonomy is optional; regex fallback still works


# ──────────────────────────────────────────────
# LLM-based parser (primary, falls back on failure)
# ──────────────────────────────────────────────

def _llm_parse_jd(text: str) -> Optional[Dict[str, Any]]:
    """
    Use an LLM to extract structured JD data.
    Returns None if no LLM API key is configured or on any error.
    """
    prompt = f"""You are an expert HR analyst. Parse the following Job Description into structured JSON.

Return ONLY valid JSON with these exact keys (no extra text):
{{
  "title": "Job title string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["optional_skill1"],
  "experience_years": {{"min": 0, "max": 10}},
  "experience_level": "Junior|Mid|Senior|Lead|Principal",
  "education": "Full education requirement phrase",
  "education_level": "high_school|associate|bachelor|master|phd",
  "keywords": ["keyword1", "keyword2"],
  "domain": "Primary domain e.g. Machine Learning, Web Development, Data Engineering",
  "responsibilities": ["responsibility sentence 1", "responsibility sentence 2"]
}}

Job Description:
\"\"\"
{text[:5000]}
\"\"\""""

    # ── Try OpenAI ──────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            pass

    # ── Try Gemini ──────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception:
            pass

    # ── Try Anthropic Claude ─────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception:
            pass

    return None  # No LLM available


# ──────────────────────────────────────────────
# Rule-based fallback parser
# ──────────────────────────────────────────────

def _rule_based_parse(text: str) -> Dict[str, Any]:
    """Offline rule-based JD parser using regex + skill taxonomy."""
    _load_taxonomy()

    title = _extract_job_title(text)
    required_skills, preferred_skills = _extract_skills(text)
    exp_min, exp_max = _extract_experience_years(text)
    exp_level = _infer_experience_level(exp_min, text)
    education, edu_level = _extract_education(text)
    keywords = _extract_keywords(text)
    domain = _infer_domain(required_skills + preferred_skills, text)
    responsibilities = _extract_responsibilities(text)

    return {
        "title": title,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_years": {"min": exp_min, "max": exp_max},
        "experience_level": exp_level,
        "education": education,
        "education_level": edu_level,
        "keywords": keywords,
        "domain": domain,
        "responsibilities": responsibilities,
    }


def _extract_job_title(text: str) -> str:
    # Look for explicit label first
    match = re.search(r"(?i)(?:job\s*title|position|role)\s*[:\-]\s*(.+)", text)
    if match:
        return match.group(1).strip()
    # Otherwise use the first non-empty line
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "Unknown Position"


def _get_section(text: str, headers: List[str]) -> str:
    """Extract the text block under any of the given section headers."""
    pattern = (
        r"(?i)(?:" + "|".join(re.escape(h) for h in headers) + r")"
        r"[^\n]*\n([\s\S]+?)(?=\n[A-Z][A-Z\s]{3,}\n|$)"
    )
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _extract_skills(text: str) -> Tuple[List[str], List[str]]:
    _load_taxonomy()
    text_lower = text.lower()

    all_skills: List[str] = []
    for category, skills in _SKILL_TAXONOMY.items():
        all_skills.extend(skills)

    req_section = _get_section(text, ["required", "must have", "mandatory", "qualifications"])
    pref_section = _get_section(text, ["preferred", "nice to have", "bonus", "plus", "desired"])

    required: List[str] = []
    preferred: List[str] = []

    for skill in all_skills:
        skill_lower = skill.lower()
        if not re.search(rf"\b{re.escape(skill_lower)}\b", text_lower):
            continue
        if req_section and re.search(rf"\b{re.escape(skill_lower)}\b", req_section.lower()):
            required.append(skill)
        elif pref_section and re.search(rf"\b{re.escape(skill_lower)}\b", pref_section.lower()):
            preferred.append(skill)
        else:
            required.append(skill)

    return list(dict.fromkeys(required)), list(dict.fromkeys(preferred))


def _extract_experience_years(text: str) -> Tuple[int, int]:
    patterns = re.compile(
        r"(\d+)\s*(?:to|\-|–)\s*(\d+)\s*years?|"
        r"(\d+)\+?\s*years?|"
        r"minimum\s+of\s+(\d+)\s*years?",
        re.IGNORECASE,
    )
    for m in patterns.finditer(text):
        if m.group(1) and m.group(2):
            return int(m.group(1)), int(m.group(2))
        val = int(m.group(3) or m.group(4) or 0)
        return val, val + 3
    return 0, 10


def _infer_experience_level(exp_min: int, text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["principal", "staff engineer", "distinguished"]):
        return "Principal"
    if any(w in t for w in ["lead", "tech lead", "team lead"]):
        return "Lead"
    if any(w in t for w in ["senior", "sr."]):
        return "Senior"
    if any(w in t for w in ["junior", "jr.", "entry level", "entry-level"]):
        return "Junior"
    if exp_min >= 7:
        return "Senior"
    if exp_min >= 3:
        return "Mid"
    return "Junior"


def _extract_education(text: str) -> Tuple[str, str]:
    _load_taxonomy()
    text_lower = text.lower()
    for edu_key, level in sorted(_EDUCATION_LEVELS.items(), key=lambda x: -x[1]):
        if edu_key in text_lower:
            match = re.search(rf"({re.escape(edu_key)}[^\n.;]{{0,60}})", text_lower)
            full = match.group(1).strip() if match else edu_key
            return full.title(), edu_key
    return "Not specified", "any"


def _extract_keywords(text: str) -> List[str]:
    """Extract high-frequency capitalised terms as keywords."""
    words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:20]]


def _infer_domain(skills: List[str], text: str) -> str:
    skill_set = {s.lower() for s in skills}
    t = text.lower()

    domain_signals: Dict[str, set] = {
        "Machine Learning / AI": {
            "machine learning", "deep learning", "pytorch", "tensorflow",
            "nlp", "llm", "ai", "bert", "gpt", "transformers",
        },
        "Data Engineering": {
            "spark", "kafka", "airflow", "etl", "data pipeline",
            "dbt", "databricks", "flink",
        },
        "Data Science": {
            "pandas", "numpy", "sklearn", "scikit-learn", "r",
            "statistics", "analytics", "jupyter",
        },
        "Cloud / DevOps": {
            "aws", "gcp", "azure", "docker", "kubernetes",
            "terraform", "ci/cd", "helm",
        },
        "Web Development": {
            "react", "vue", "angular", "node.js", "django",
            "fastapi", "flask", "next.js",
        },
        "Mobile Development": {
            "swift", "kotlin", "flutter", "react native",
            "ios", "android",
        },
        "Cybersecurity": {
            "security", "penetration testing", "siem", "soc",
            "encryption", "firewall", "zero trust",
        },
    }

    scores = {
        domain: sum(1 for s in signals if s in skill_set or s in t)
        for domain, signals in domain_signals.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Software Engineering"


def _extract_responsibilities(text: str) -> List[str]:
    section = _get_section(
        text, ["responsibilities", "duties", "what you'll do", "what you will do", "role"]
    )
    if not section:
        return []
    lines = [
        l.strip().lstrip("•·-*–") for l in section.splitlines() if l.strip()
    ]
    return [l for l in lines if len(l) > 10][:15]


# ──────────────────────────────────────────────
# Public Agent class
# ──────────────────────────────────────────────

class JDParserAgent:
    """
    Parses a Job Description string into structured metadata.

    Strategy:
      1. Try LLM (if API key present) for richest extraction.
      2. Fall back to regex + taxonomy-based rule parser (fully offline).

    Usage:
        agent = JDParserAgent(use_llm=True)
        jd = agent.parse(jd_text_string)
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def parse(self, jd_text: str) -> Dict[str, Any]:
        """
        Parse a JD text and return a structured dict.

        Args:
            jd_text : raw job description string

        Returns:
            Structured JD dict with all required keys.
        """
        result: Optional[Dict[str, Any]] = None

        if self.use_llm:
            result = _llm_parse_jd(jd_text)

        if result is None:
            # Offline fallback — always reliable
            result = _rule_based_parse(jd_text)

        # Always attach raw text for downstream semantic matching
        result["raw_text"] = jd_text
        return result
