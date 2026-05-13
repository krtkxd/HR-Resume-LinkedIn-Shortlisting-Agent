"""
agents/profile_parser.py
────────────────────────
Profile Parser Agent

Converts raw resume text (or LinkedIn dict) into a structured candidate profile:
  {
    "candidate_id": str,
    "name": str,
    "email": str,
    "phone": str,
    "summary": str,
    "skills": [...],
    "experience": [{"title","company","duration","description","years"}],
    "education": [{"degree","field","school","year"}],
    "certifications": [...],
    "projects": [...],
    "total_experience_years": float,
    "highest_education_level": str,
    "raw_text": str
  }

Uses LLM with regex fallback.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# LLM profile extraction
# ──────────────────────────────────────────────

def _llm_parse_profile(text: str) -> Optional[Dict[str, Any]]:
    prompt = f"""Extract structured information from this resume/profile text.
Return ONLY valid JSON with these exact keys:
{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1234567890",
  "summary": "Professional summary in 2-3 sentences",
  "skills": ["skill1", "skill2"],
  "experience": [
    {{"title": "Job Title", "company": "Company Name", "duration": "Jan 2020 - Dec 2022", "description": "What they did", "years": 2.0}}
  ],
  "education": [
    {{"degree": "Bachelor of Science", "field": "Computer Science", "school": "MIT", "year": "2018"}}
  ],
  "certifications": ["AWS Certified Solutions Architect"],
  "projects": ["Project name: description"]
}}

Resume Text:
{text[:4000]}"""

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

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            raw = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception:
            pass

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

    return None


# ──────────────────────────────────────────────
# Rule-based fallback
# ──────────────────────────────────────────────

_SKILL_PATTERNS: List[str] = []

def _load_skills() -> None:
    global _SKILL_PATTERNS
    if _SKILL_PATTERNS:
        return
    skills_path = os.path.join(os.path.dirname(__file__), "..", "data", "skills.json")
    try:
        with open(skills_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if isinstance(v, list):
                _SKILL_PATTERNS.extend(v)
    except Exception:
        pass


def _rule_based_parse(text: str) -> Dict[str, Any]:
    _load_skills()
    name = _extract_name(text)
    email = _extract_email(text)
    phone = _extract_phone(text)
    skills = _extract_skills(text)
    experience = _extract_experience(text)
    education = _extract_education(text)
    certifications = _extract_certifications(text)
    projects = _extract_projects(text)
    summary = _extract_summary(text)
    total_years = _compute_total_experience(experience)
    edu_level = _highest_education(education)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "summary": summary,
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "projects": projects,
        "total_experience_years": total_years,
        "highest_education_level": edu_level,
    }


def _extract_name(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Name is usually the first line that looks like proper words (no special chars)
    for line in lines[:5]:
        if re.match(r"^[A-Z][a-zA-Z\s\-'\.]{2,40}$", line):
            return line
    return lines[0] if lines else "Unknown"


def _extract_email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else ""


def _extract_phone(text: str) -> str:
    match = re.search(r"[\+\(]?[\d\s\-\(\)]{10,18}", text)
    return match.group(0).strip() if match else ""


def _extract_skills(text: str) -> List[str]:
    _load_skills()
    text_lower = text.lower()
    found = []
    for skill in _SKILL_PATTERNS:
        if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))


def _extract_summary(text: str) -> str:
    for header in ["summary", "objective", "profile", "about"]:
        pattern = rf"(?i){header}\s*\n([\s\S]{{30,500}}?)(?=\n[A-Z]{{2,}}|\n\n[A-Z]|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    # Use first paragraph as fallback
    paras = re.split(r"\n{2,}", text)
    for p in paras[1:4]:
        if len(p.strip()) > 50:
            return p.strip()
    return ""


def _extract_experience(text: str) -> List[Dict[str, Any]]:
    blocks = []
    # Date range pattern
    date_pattern = re.compile(
        r"(?P<title>[A-Z][^\n]{5,60})\n"
        r"(?P<company>[^\n]{2,60})\n"
        r"(?P<dates>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|"
        r"March|April|June|July|August|September|October|November|December)?\s*\d{4}"
        r"\s*[–\-]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|"
        r"February|March|April|June|July|August|September|October|November|December)"
        r"?\s*\d{4}|Present|Current))",
        re.MULTILINE,
    )
    for m in date_pattern.finditer(text):
        years = _parse_duration_years(m.group("dates"))
        blocks.append({
            "title": m.group("title").strip(),
            "company": m.group("company").strip(),
            "duration": m.group("dates").strip(),
            "description": "",
            "years": years,
        })
    return blocks


def _parse_duration_years(duration: str) -> float:
    """Estimate years from a duration string like 'Jan 2020 - Dec 2022'."""
    years_found = re.findall(r"\d{4}", duration)
    if len(years_found) >= 2:
        return abs(int(years_found[1]) - int(years_found[0]))
    if "present" in duration.lower() or "current" in duration.lower():
        import datetime
        start_year = int(years_found[0]) if years_found else 2020
        return datetime.datetime.now().year - start_year
    return 1.0


def _compute_total_experience(experience: List[Dict]) -> float:
    return round(sum(e.get("years", 0) for e in experience), 1)


def _extract_education(text: str) -> List[Dict[str, str]]:
    degrees = ["phd", "doctorate", "master", "msc", "mtech", "mba",
               "bachelor", "bsc", "btech", "b.e", "be ", "associate", "diploma"]
    edu_list = []
    for deg in degrees:
        pattern = re.compile(
            rf"(?i)({deg}[^\n]{{0,80}})\n([^\n]{{2,80}})",
        )
        for m in pattern.finditer(text):
            edu_list.append({
                "degree": m.group(1).strip(),
                "field": "",
                "school": m.group(2).strip(),
                "year": "",
            })
    return edu_list[:5]


def _highest_education(education: List[Dict]) -> str:
    edu_rank = {"phd": 5, "doctorate": 5, "master": 4, "msc": 4,
                "mtech": 4, "mba": 4, "bachelor": 3, "bsc": 3,
                "btech": 3, "associate": 2, "diploma": 2}
    best = "unknown"
    best_rank = 0
    for e in education:
        deg = e.get("degree", "").lower()
        for key, rank in edu_rank.items():
            if key in deg and rank > best_rank:
                best_rank = rank
                best = key
    return best


def _extract_certifications(text: str) -> List[str]:
    section = _get_section(text, ["certifications", "licenses", "courses"])
    if not section:
        return []
    raw = re.split(r"[•·\n]+", section)
    return [s.strip() for s in raw if 3 < len(s.strip()) < 100][:10]


def _extract_projects(text: str) -> List[str]:
    section = _get_section(text, ["projects", "portfolio", "works"])
    if not section:
        return []
    raw = re.split(r"[•·\n]+", section)
    return [s.strip() for s in raw if len(s.strip()) > 10][:10]


def _get_section(text: str, headers: List[str]) -> str:
    pattern = r"(?i)(?:" + "|".join(re.escape(h) for h in headers) + r")[^\n]*\n([\s\S]+?)(?=\n[A-Z][A-Z\s]{3,}\n|$)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""


# ──────────────────────────────────────────────
# Public agent
# ──────────────────────────────────────────────

class ProfileParserAgent:
    """
    Parses a raw resume text (or LinkedIn dict) into a structured candidate profile.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def parse_resume(self, text: str, filename: str = "") -> Dict[str, Any]:
        result = None
        if self.use_llm:
            result = _llm_parse_profile(text)

        if result is None:
            result = _rule_based_parse(text)

        result["candidate_id"] = str(uuid.uuid4())[:8]
        result["source"] = filename or "resume"
        result["raw_text"] = text
        # Ensure computed fields exist
        if "total_experience_years" not in result:
            result["total_experience_years"] = _compute_total_experience(
                result.get("experience", [])
            )
        if "highest_education_level" not in result:
            result["highest_education_level"] = _highest_education(
                result.get("education", [])
            )
        return result

    def parse_linkedin(self, linkedin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert already-parsed LinkedIn dict to candidate profile schema."""
        raw_text = linkedin_data.get("raw_text", json.dumps(linkedin_data))
        result = self.parse_resume(raw_text, filename="linkedin")
        # Prefer LinkedIn-specific fields when available
        if linkedin_data.get("name"):
            result["name"] = linkedin_data["name"]
        if linkedin_data.get("skills"):
            # Merge and deduplicate
            merged = list(dict.fromkeys(
                linkedin_data["skills"] + result.get("skills", [])
            ))
            result["skills"] = merged
        result["headline"] = linkedin_data.get("headline", "")
        return result
