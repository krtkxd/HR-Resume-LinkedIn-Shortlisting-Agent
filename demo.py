"""
demo.py
────────
End-to-end demonstration of the HR Shortlisting Agent pipeline.
Run from the project root:  python demo.py

Reads sample JD + 3 resumes from samples/ directory.
Outputs JSON + HTML report to outputs/ directory.

Run from the project root:
    python demo.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# ── Fix Windows console UTF-8 (emoji support) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── ensure imports work from project root ────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.jd_parser import JDParserAgent
from agents.profile_parser import ProfileParserAgent
from agents.ranking_agent import RankingAgent
from agents.scoring_agent import ScoringAgent
from report.generator import ReportGenerator
from utils.resume_parser import extract_text


def main():
    print("=" * 60)
    print("  HR Resume & LinkedIn Shortlisting Agent — Demo")
    print("=" * 60)

    SAMPLES = ROOT / "samples"
    OUTPUT  = ROOT / "outputs"
    OUTPUT.mkdir(exist_ok=True)

    # ── 1. Load JD ───────────────────────────
    jd_file = SAMPLES / "sample_jd.txt"
    if not jd_file.exists():
        print("❌  Sample JD not found. Ensure samples/sample_jd.txt exists.")
        sys.exit(1)

    jd_text = jd_file.read_text(encoding="utf-8")
    print(f"\n📋 Job Description loaded ({len(jd_text)} chars).\n")

    # ── 2. Init agents ───────────────────────
    print("🤖 Initialising agents...")
    use_llm = bool(
        os.getenv("OPENAI_API_KEY") or
        os.getenv("GEMINI_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )
    print(f"   LLM mode: {'ON (API key found)' if use_llm else 'OFF (using rule-based fallback)'}")
    print("   Embedding backend: sentence_transformers")

    jd_agent      = JDParserAgent(use_llm=use_llm)
    profile_agent = ProfileParserAgent(use_llm=use_llm)
    scoring_agent = ScoringAgent(embedding_backend="sentence_transformers", use_llm=use_llm)
    ranking_agent = RankingAgent()
    report_gen    = ReportGenerator()

    # ── 3. Parse JD ──────────────────────────
    print("\n🔍 Parsing Job Description...")
    parsed_jd = jd_agent.parse(jd_text)
    print(f"   Title         : {parsed_jd.get('title')}")
    print(f"   Domain        : {parsed_jd.get('domain')}")
    print(f"   Exp required  : {parsed_jd.get('experience_years')}")
    print(f"   Edu required  : {parsed_jd.get('education_level')}")
    print(f"   Required skills ({len(parsed_jd.get('required_skills',[]))}): "
          f"{', '.join(parsed_jd.get('required_skills',[])[:8])}...")

    # ── 4. Parse resumes ─────────────────────
    resume_files = sorted(SAMPLES.glob("resume_*.txt"))
    if not resume_files:
        print("❌  No resume files found in samples/")
        sys.exit(1)

    print(f"\n📂 Processing {len(resume_files)} resume(s)...")
    candidates = []
    for rf in resume_files:
        raw_text = rf.read_text(encoding="utf-8")
        profile = profile_agent.parse_resume(raw_text, filename=rf.name)
        candidates.append(profile)
        print(f"   ✅ {rf.name} → {profile.get('name')} | "
              f"Skills: {len(profile.get('skills',[]))} | "
              f"Exp: {profile.get('total_experience_years')} yrs | "
              f"Edu: {profile.get('highest_education_level')}")

    # ── 5. Score candidates ──────────────────
    print("\n📊 Scoring candidates against JD rubric...")
    scored = []
    for c in candidates:
        result = scoring_agent.score(c, parsed_jd)
        scored.append(result)
        print(f"   {result['name']:25s} | "
              f"Skills:{result['scores']['skills_match']['score']:2d} "
              f"Exp:{result['scores']['experience']['score']:2d} "
              f"Edu:{result['scores']['education']['score']:2d} "
              f"Proj:{result['scores']['projects']['score']:2d} "
              f"Comm:{result['scores']['communication']['score']:2d} "
              f"| Total: {result['total_score']:.2f} "
              f"| {result['recommendation']}")

    # ── 6. Rank ───────────────────────────────
    print("\n🏆 Ranking candidates...")
    ranked = ranking_agent.rank(scored)
    stats  = ranking_agent.summary_stats(ranked)

    print("\n  Final Ranking:")
    print("  " + "-" * 55)
    for c in ranked:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(c["rank"], f"  #{c['rank']}")
        print(f"  {medal}  {c['name']:25s} {c['total_score']:.2f}/10  "
              f"Tier {c['tier']}  {c['recommendation']}")
    print("  " + "-" * 55)
    print(f"\n  Average score   : {stats['average_score']:.2f}")
    print(f"  Hire candidates : {stats['hire_count']}")

    # ── 7. Demo override ─────────────────────
    print("\n✏️  Demo: Applying manual override for last candidate...")
    last = ranked[-1]
    scoring_agent.apply_override(
        result=last,
        category="projects",
        new_score=7,
        reason="Reviewed GitHub — has unreported personal ML projects",
        overrider="Demo HR Manager",
    )
    ranked = ranking_agent.rank(ranked)
    print(f"   Override applied. New total: {last['total_score']:.2f}")

    # ── 8. Generate reports ───────────────────
    print("\n📄 Generating reports...")

    json_path = OUTPUT / "report.json"
    json_str = report_gen.generate_json(ranked, parsed_jd, stats, str(json_path))
    print(f"   JSON report → {json_path}")

    html_path = OUTPUT / "report.html"
    report_gen.generate_html(ranked, parsed_jd, stats, str(html_path))
    print(f"   HTML report → {html_path}")

    # Try PDF
    try:
        pdf_path = OUTPUT / "report.pdf"
        report_gen.generate_pdf(ranked, parsed_jd, stats, str(pdf_path))
        print(f"   PDF  report → {pdf_path}")
    except ImportError:
        print("   PDF skipped (reportlab not installed). Run: pip install reportlab")

    # ── 9. Print sample JSON output ──────────
    print("\n📌 Sample JSON output (first candidate):")
    top = ranked[0]
    sample_output = {
        "name": top["name"],
        "scores": {
            k: {
                "score": v["score"],
                "weight": v["weight"],
                "justification": v["justification"],
            }
            for k, v in top["scores"].items()
        },
        "total_score": top["total_score"],
        "recommendation": top["recommendation"],
    }
    print(json.dumps(sample_output, indent=2))

    print("\n✅  Demo complete! Open outputs/report.html in a browser to view the full report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
