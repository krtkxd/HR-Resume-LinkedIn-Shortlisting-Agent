"""
utils/linkedin_parser.py
────────────────────────
Parses LinkedIn profile data in two formats:
  1. Raw text (copy-pasted from LinkedIn)
  2. JSON (exported via LinkedIn Data Export or scraped)

Normalises both into the same structured dict schema used by profile_parser.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────

def parse_linkedin(raw: str | dict) -> Dict[str, Any]:
    """
    Parse a LinkedIn profile from either raw text or a dict/JSON string.

    Returns a dict with keys:
        name, headline, location, summary, skills,
        experience, education, certifications, projects
    """
    if isinstance(raw, str):
        # Try JSON parse first
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                data = json.loads(stripped)
                return _parse_json_profile(data)
            except json.JSONDecodeError:
                pass
        # Fall back to free-text parsing
        return _parse_text_profile(raw)
    elif isinstance(raw, dict):
        return _parse_json_profile(raw)
    else:
        raise ValueError("Input must be a string (JSON or plain text) or a dict.")


# ──────────────────────────────────────────────
# JSON profile parser
# ──────────────────────────────────────────────

def _parse_json_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle LinkedIn's export JSON format.
    Also handles common scraping output schemas.
    """
    # Try to extract experience blocks
    experience = []
    for exp in data.get("experience", data.get("positions", [])):
        experience.append({
            "title": exp.get("title", exp.get("role", "")),
            "company": exp.get("company", exp.get("companyName", "")),
            "duration": exp.get("duration", exp.get("timePeriod", "")),
            "description": exp.get("description", ""),
        })

    # Education blocks
    education = []
    for edu in data.get("education", []):
        education.append({
            "degree": edu.get("degree", edu.get("degreeName", "")),
            "field": edu.get("fieldOfStudy", edu.get("field", "")),
            "school": edu.get("school", edu.get("schoolName", "")),
            "year": str(edu.get("endDate", edu.get("year", ""))),
        })

    # Skills
    skills_raw = data.get("skills", [])
    skills = []
    for s in skills_raw:
        if isinstance(s, str):
            skills.append(s)
        elif isinstance(s, dict):
            skills.append(s.get("name", s.get("skill", "")))

    # Certifications
    certs = []
    for c in data.get("certifications", data.get("courses", [])):
        if isinstance(c, str):
            certs.append(c)
        elif isinstance(c, dict):
            certs.append(c.get("name", c.get("title", "")))

    return {
        "name": data.get("name", data.get("firstName", "") + " " + data.get("lastName", "")).strip(),
        "headline": data.get("headline", data.get("title", "")),
        "location": data.get("location", data.get("geoLocation", "")),
        "summary": data.get("summary", data.get("about", "")),
        "skills": [s for s in skills if s],
        "experience": experience,
        "education": education,
        "certifications": [c for c in certs if c],
        "projects": data.get("projects", []),
        "raw_text": json.dumps(data, indent=2),
    }


# ──────────────────────────────────────────────
# Free-text profile parser
# ──────────────────────────────────────────────

def _parse_text_profile(text: str) -> Dict[str, Any]:
    """
    Heuristically extract sections from pasted LinkedIn profile text.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    name = _extract_name(lines)
    headline = lines[1] if len(lines) > 1 else ""
    summary = _extract_section(text, ["about", "summary"])
    skills = _extract_skills_from_text(text)
    experience = _extract_experience_blocks(text)
    education = _extract_education_blocks(text)
    certs = _extract_certifications(text)

    return {
        "name": name,
        "headline": headline,
        "location": _extract_location(text),
        "summary": summary,
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certs,
        "projects": [],
        "raw_text": text,
    }


def _extract_name(lines: List[str]) -> str:
    # First non-empty line is usually the name
    return lines[0] if lines else "Unknown"


def _extract_location(text: str) -> str:
    # Patterns like "San Francisco, CA" or "New York, United States"
    match = re.search(r"([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){1,3})", text[:500])
    return match.group(1) if match else ""


def _extract_section(text: str, section_names: List[str]) -> str:
    """Extract text under a section header."""
    pattern = r"(?i)(?:" + "|".join(section_names) + r")\s*\n([\s\S]+?)(?=\n[A-Z][A-Z\s]{3,}\n|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_skills_from_text(text: str) -> List[str]:
    """Extract skill bullets from a Skills section."""
    section = _extract_section(text, ["skills", "top skills"])
    if not section:
        return []
    # Skills are often bullet-separated or newline-separated
    raw = re.split(r"[•·\|\n,;]+", section)
    return [s.strip() for s in raw if 2 < len(s.strip()) < 60]


def _extract_experience_blocks(text: str) -> List[Dict[str, str]]:
    """Very rough heuristic: capture title + company + date pattern."""
    pattern = re.compile(
        r"(?P<title>[A-Z][^\n]{3,60})\n"
        r"(?P<company>[^\n]{2,60})\n"
        r"(?P<dates>\w+ \d{4}\s*[–\-]\s*(?:\w+ \d{4}|Present))",
        re.MULTILINE,
    )
    blocks = []
    for m in pattern.finditer(text):
        blocks.append({
            "title": m.group("title").strip(),
            "company": m.group("company").strip(),
            "duration": m.group("dates").strip(),
            "description": "",
        })
    return blocks


def _extract_education_blocks(text: str) -> List[Dict[str, str]]:
    """Heuristic: look for degree keywords."""
    degrees = [
        "Bachelor", "Master", "PhD", "Doctorate", "B.Tech", "M.Tech",
        "B.Sc", "M.Sc", "MBA", "B.E.", "M.E.",
    ]
    edu_list = []
    for deg in degrees:
        pattern = re.compile(rf"({deg}[^\n]{{0,80}})\n([^\n]{{2,80}})", re.IGNORECASE)
        for m in pattern.finditer(text):
            edu_list.append({
                "degree": m.group(1).strip(),
                "field": "",
                "school": m.group(2).strip(),
                "year": "",
            })
    return edu_list


def _extract_certifications(text: str) -> List[str]:
    """Extract certification names from a Certifications section."""
    section = _extract_section(text, ["certifications", "licenses & certifications", "courses"])
    if not section:
        return []
    raw = re.split(r"[•·\n]+", section)
    return [s.strip() for s in raw if 3 < len(s.strip()) < 100]
