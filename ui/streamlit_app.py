"""
ui/streamlit_app.py
────────────────────
Streamlit UI for the HR Shortlisting Agent.
Run with: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Path setup ───────────────────────────────
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
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="HR Shortlisting Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0f1117; }
.hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px; padding: 32px 40px; margin-bottom: 24px; text-align: center; }
.hero h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { color: rgba(255,255,255,0.8); margin-top: 8px; font-size: 1rem; }
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.metric-card { flex: 1; min-width: 120px; background: #1e2030;
  border: 1px solid #2d3148; border-radius: 12px; padding: 18px 20px; }
.metric-card .num { font-size: 2rem; font-weight: 700; color: #818cf8; }
.metric-card .lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 20px;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
.pill-sh { background:#065f46; color:#34d399; }
.pill-h  { background:#1e3a5f; color:#60a5fa; }
.pill-m  { background:#713f12; color:#fbbf24; }
.pill-nh { background:#4c1d1d; color:#f87171; }
.sbar-wrap { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.sbar-label { font-size: 0.72rem; color: #64748b; width: 110px; text-align: right; }
.sbar { flex: 1; height: 6px; background: #2d3148; border-radius: 3px; overflow: hidden; }
.sbar-fill { height: 100%; border-radius: 3px; }
.sbar-val { font-size: 0.78rem; font-weight: 600; width: 28px; }
.gap-tag { display: inline-block; padding: 2px 8px; margin: 2px;
  border-radius: 4px; background: #4c1d1d; color: #fca5a5; font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state init  (BEFORE any widgets)
# ──────────────────────────────────────────────

SAMPLE_JD_PATH = ROOT / "samples" / "sample_jd.txt"
SAMPLE_RESUME_DIR = ROOT / "samples"

# Pre-populate JD text on first load from sample file
if "jd_prefill" not in st.session_state:
    if SAMPLE_JD_PATH.exists():
        st.session_state["jd_prefill"] = SAMPLE_JD_PATH.read_text(encoding="utf-8")
    else:
        st.session_state["jd_prefill"] = ""

if "ranked" not in st.session_state:
    st.session_state.ranked = []
if "jd" not in st.session_state:
    st.session_state.jd = {}
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_agents(backend: str, use_llm: bool):
    return (
        JDParserAgent(use_llm=use_llm),
        ProfileParserAgent(use_llm=use_llm),
        ScoringAgent(embedding_backend=backend, use_llm=use_llm),
        RankingAgent(),
        ReportGenerator(),
    )


def score_color(s: int) -> str:
    if s >= 8: return "#34d399"
    if s >= 5: return "#60a5fa"
    return "#f87171"


def pill_class(rec: str) -> str:
    return {
        "Strong Hire": "pill-sh",
        "Hire": "pill-h",
        "Maybe": "pill-m",
        "No Hire": "pill-nh",
    }.get(rec, "pill-m")


def run_pipeline_on_samples(jd_text: str, backend: str, use_llm: bool, top_n: int):
    """Run full pipeline on the bundled sample resumes."""
    jd_ag, prof_ag, score_ag, rank_ag, rep_gen = load_agents(backend, use_llm)

    parsed_jd = jd_ag.parse(jd_text)

    candidates = []
    for resume_file in sorted(SAMPLE_RESUME_DIR.glob("resume_*.txt")):
        raw_text = resume_file.read_text(encoding="utf-8")
        profile = prof_ag.parse_resume(raw_text, filename=resume_file.name)
        candidates.append(profile)

    scored = [score_ag.score(c, parsed_jd) for c in candidates]
    ranked = rank_ag.rank(scored, top_n=top_n)
    stats  = rank_ag.summary_stats(ranked)

    st.session_state.ranked          = ranked
    st.session_state.jd              = parsed_jd
    st.session_state.stats           = stats
    st.session_state._report_gen     = rep_gen
    st.session_state._scoring_agent  = score_ag
    st.session_state._ranking_agent  = rank_ag


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    embedding_backend = st.selectbox(
        "Embedding Backend",
        ["sentence_transformers", "openai", "gemini"],
        index=0,
    )

    use_llm = st.toggle("Use LLM for Parsing", value=False,
        help="Needs an API key. Disable for fully offline rule-based mode.")

    if use_llm:
        with st.expander("🔑 API Keys (optional)"):
            oai_key = st.text_input("OpenAI API Key", type="password")
            gem_key = st.text_input("Gemini API Key", type="password")
            ant_key = st.text_input("Anthropic API Key", type="password")
            if oai_key: os.environ["OPENAI_API_KEY"] = oai_key
            if gem_key: os.environ["GEMINI_API_KEY"] = gem_key
            if ant_key: os.environ["ANTHROPIC_API_KEY"] = ant_key

    st.markdown("---")
    st.markdown("### 📊 Scoring Weights")
    st.markdown("""
| Category | Weight |
|----------|--------|
| Skills Match | **30%** |
| Experience | **25%** |
| Education | **15%** |
| Projects | **20%** |
| Communication | **10%** |
""")
    st.markdown("---")
    st.markdown("**Recommendation Thresholds:**")
    st.markdown("- 🟢 Strong Hire: ≥ 8.0")
    st.markdown("- 🔵 Hire: ≥ 6.5")
    st.markdown("- 🟡 Maybe: ≥ 5.0")
    st.markdown("- 🔴 No Hire: < 5.0")

# ──────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🎯 HR Resume & LinkedIn Shortlisting Agent</h1>
  <p>AI-powered candidate evaluation · Semantic scoring · Explainable results</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab_input, tab_results, tab_override, tab_report = st.tabs([
    "📥 Input", "📊 Results", "✏️ Override", "📄 Report"
])

# ──────────────────────────────────────────────
# INPUT TAB
# ──────────────────────────────────────────────

with tab_input:

    # ── Quick Demo Banner ─────────────────────
    st.info(
        "⚡ **Quick Demo:** Click **'Run Sample Analysis'** below to instantly analyze "
        "3 built-in resumes against the sample ML Engineer JD — no uploads needed!"
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📋 Job Description")
        jd_text = st.text_area(
            "Paste the full Job Description here:",
            value=st.session_state["jd_prefill"],
            height=350,
            placeholder="Senior Machine Learning Engineer\n\nWe are looking for...",
        )

        st.markdown("### 🔗 LinkedIn Profile (Optional)")
        linkedin_text = st.text_area(
            "Paste LinkedIn profile text or JSON export:",
            height=150,
            placeholder='{"name": "Jane Doe", "skills": ["Python", "ML"]}',
        )

    with col2:
        st.markdown("### 📂 Resume Files")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX resumes (multiple allowed):",
            type=["pdf", "docx", "doc", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for f in uploaded_files:
                st.success(f"✅ {f.name} ({f.size // 1024} KB)")

        st.markdown("### 🚀 Run Analysis")
        top_n = st.slider("Show Top N candidates:", 1, 20, 10)

        # ── Primary: Run on uploaded files ────
        run_btn = st.button(
            "▶  Analyze Candidates",
            type="primary",
            use_container_width=True,
            help="Upload resumes above first",
        )

        st.markdown("---")

        # ── Sample analysis (no upload needed) ─
        sample_btn = st.button(
            "⚡ Run Sample Analysis (3 built-in resumes)",
            use_container_width=True,
            type="secondary",
        )

    # ── Handle Sample Run ─────────────────────
    if sample_btn:
        jd_src = jd_text.strip() or st.session_state["jd_prefill"]
        if not jd_src.strip():
            st.error("No JD text found. Please paste a job description.")
        else:
            with st.spinner("🧠 Running AI pipeline on sample resumes..."):
                progress = st.progress(0, text="Loading AI agents...")
                run_pipeline_on_samples(jd_src, embedding_backend, use_llm, top_n)
                progress.progress(100, text="Done!")
                time.sleep(0.4)
                progress.empty()
            n = len(st.session_state.ranked)
            st.success(f"✅ Done! Evaluated {n} sample candidate(s). Switch to the **📊 Results** tab.")

    # ── Handle Uploaded Files Run ─────────────
    if run_btn:
        if not jd_text.strip():
            st.error("Please enter a Job Description.")
        elif not uploaded_files and not (linkedin_text and linkedin_text.strip()):
            st.error("Please upload at least one resume or enter a LinkedIn profile.")
        else:
            jd_ag, prof_ag, score_ag, rank_ag, rep_gen = load_agents(embedding_backend, use_llm)
            with st.spinner("🧠 Running AI pipeline..."):
                progress = st.progress(0, text="Loading agents...")
                parsed_jd = jd_ag.parse(jd_text)

                candidates = []
                n_files = len(uploaded_files)
                for i, upload in enumerate(uploaded_files):
                    progress.progress(
                        15 + int(40 * i / max(n_files, 1)),
                        text=f"Parsing resume: {upload.name}..."
                    )
                    raw_bytes = upload.read()
                    try:
                        text = extract_text(raw_bytes, filename=upload.name)
                        profile = prof_ag.parse_resume(text, filename=upload.name)
                        candidates.append(profile)
                    except Exception as e:
                        st.warning(f"Could not parse {upload.name}: {e}")

                if linkedin_text and linkedin_text.strip():
                    progress.progress(60, text="Parsing LinkedIn profile...")
                    try:
                        li_data = parse_linkedin(linkedin_text)
                        profile = prof_ag.parse_linkedin(li_data)
                        candidates.append(profile)
                    except Exception as e:
                        st.warning(f"LinkedIn parsing error: {e}")

                progress.progress(65, text="Scoring candidates...")
                scored = [score_ag.score(c, parsed_jd) for c in candidates]
                progress.progress(85, text="Ranking...")
                ranked = rank_ag.rank(scored, top_n=top_n)
                stats  = rank_ag.summary_stats(ranked)
                progress.progress(100, text="Done!")
                time.sleep(0.3)
                progress.empty()

                st.session_state.ranked         = ranked
                st.session_state.jd             = parsed_jd
                st.session_state.stats          = stats
                st.session_state._report_gen    = rep_gen
                st.session_state._scoring_agent = score_ag
                st.session_state._ranking_agent = rank_ag

            st.success(f"✅ Analysis complete! Evaluated {len(scored)} candidate(s). Go to **📊 Results**.")


# ──────────────────────────────────────────────
# RESULTS TAB
# ──────────────────────────────────────────────

with tab_results:
    ranked = st.session_state.ranked
    stats  = st.session_state.stats
    jd     = st.session_state.jd

    if not ranked:
        st.info("⬅️ Go to the **📥 Input** tab and click **⚡ Run Sample Analysis** to see results here.")
    else:
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card"><div class="num">{stats.get('total_candidates',0)}</div><div class="lbl">Evaluated</div></div>
          <div class="metric-card"><div class="num">{stats.get('hire_count',0)}</div><div class="lbl">Hire / Strong Hire</div></div>
          <div class="metric-card"><div class="num">{stats.get('average_score',0):.1f}</div><div class="lbl">Avg Score</div></div>
          <div class="metric-card"><div class="num">{stats.get('highest_score',0):.1f}</div><div class="lbl">Top Score</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"**Position:** `{jd.get('title','N/A')}` · "
            f"**Domain:** `{jd.get('domain','N/A')}` · "
            f"**Exp Required:** `{jd.get('experience_years',{}).get('min',0)}–"
            f"{jd.get('experience_years',{}).get('max',10)} yrs`"
        )
        st.divider()

        for c in ranked:
            col_rank, col_info = st.columns([0.05, 0.95])
            with col_rank:
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(c["rank"], f"#{c['rank']}")
                st.markdown(f"### {medal}")
            with col_info:
                rec      = c["recommendation"]
                pill_cls = pill_class(rec)
                with st.expander(
                    f"**{c['name']}** — Score: {c['total_score']:.1f}/10 · {rec} · Tier {c['tier']}",
                    expanded=c["rank"] <= 3,
                ):
                    sc   = c["scores"]
                    cats = [
                        ("skills_match", "Skills Match",    0.30),
                        ("experience",   "Experience",      0.25),
                        ("education",    "Education",       0.15),
                        ("projects",     "Projects",        0.20),
                        ("communication","Communication",   0.10),
                    ]
                    for key, label, weight in cats:
                        s     = sc[key]["score"]
                        color = score_color(s)
                        just  = sc[key]["justification"]
                        st.markdown(f"""
                        <div class="sbar-wrap">
                          <span class="sbar-label">{label} ({int(weight*100)}%)</span>
                          <div class="sbar"><div class="sbar-fill" style="width:{s*10}%;background:{color};"></div></div>
                          <span class="sbar-val" style="color:{color};">{s}</span>
                        </div>
                        <div style="font-size:0.75rem;color:#64748b;margin-left:118px;margin-bottom:8px;">{just}</div>
                        """, unsafe_allow_html=True)

                    total_color = score_color(int(c["total_score"]))
                    st.markdown(f"""
                    <div style="margin-top:12px;padding:12px;background:#252840;border-radius:8px;
                      display:flex;align-items:center;gap:16px;">
                      <span style="font-size:2rem;font-weight:700;color:{total_color};">{c['total_score']:.1f}</span>
                      <div>
                        <span class="pill {pill_cls}">{rec}</span>
                        <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">
                          Tier {c['tier']} · Percentile {c['percentile']}th · Confidence {int(c['confidence']*100)}%
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if c.get("skill_gaps"):
                        gaps_html = "".join(
                            f'<span class="gap-tag">{g}</span>' for g in c["skill_gaps"][:12]
                        )
                        st.markdown(f"""
                        <div style="margin-top:12px;">
                          <span style="font-size:0.75rem;color:#f87171;font-weight:600;">⚠ Missing Skills:</span>
                          <div style="margin-top:6px;">{gaps_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if c.get("override_log"):
                        st.markdown("**Manual Overrides:**")
                        for ov in c["override_log"]:
                            st.caption(
                                f"📝 {ov['timestamp']} — {ov['category']}: "
                                f"{ov['old_score']}→{ov['new_score']} by {ov['overrider']} ({ov['reason']})"
                            )


# ──────────────────────────────────────────────
# OVERRIDE TAB
# ──────────────────────────────────────────────

with tab_override:
    ranked = st.session_state.ranked
    if not ranked:
        st.info("Run an analysis first from the Input tab.")
    else:
        st.markdown("### ✏️ Manual Score Override")
        st.info("Override any score dimension for any candidate. Useful when you have context the AI missed.")

        cand_names  = {c["candidate_id"]: c["name"] for c in ranked}
        selected_id = st.selectbox(
            "Select Candidate:",
            options=list(cand_names.keys()),
            format_func=lambda x: cand_names[x],
        )
        selected_cand = next(c for c in ranked if c["candidate_id"] == selected_id)

        category = st.selectbox(
            "Category to Override:",
            options=["skills_match", "experience", "education", "projects", "communication"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        current_score = selected_cand["scores"][category]["score"]
        st.caption(f"Current score: **{current_score}/10**")
        st.caption(f"Justification: _{selected_cand['scores'][category]['justification']}_")

        new_score = st.slider("New Score:", 0, 10, current_score)
        reason    = st.text_input("Reason for override:", placeholder="Reviewed GitHub portfolio — strong ML projects...")
        overrider = st.text_input("Your name/role:", value="HR Manager")

        if st.button("✅ Apply Override", type="primary"):
            if not reason.strip():
                st.error("Please provide a reason for the override.")
            else:
                score_ag = st.session_state.get("_scoring_agent")
                rank_ag  = st.session_state.get("_ranking_agent")
                if score_ag:
                    score_ag.apply_override(
                        result=selected_cand,
                        category=category,
                        new_score=new_score,
                        reason=reason,
                        overrider=overrider,
                    )
                    st.session_state.ranked = rank_ag.rank(ranked)
                    st.session_state.stats  = rank_ag.summary_stats(st.session_state.ranked)
                    st.success(f"✅ Override applied! New total score: {selected_cand['total_score']:.2f}/10")
                    st.balloons()


# ──────────────────────────────────────────────
# REPORT TAB
# ──────────────────────────────────────────────

with tab_report:
    ranked = st.session_state.ranked
    jd     = st.session_state.jd
    stats  = st.session_state.stats

    if not ranked:
        st.info("Run an analysis first from the Input tab.")
    else:
        rep_gen = st.session_state.get("_report_gen", ReportGenerator())

        col_html, col_json = st.columns(2)

        with col_html:
            st.markdown("### 🖥️ HTML Report")
            if st.button("Generate HTML Report", use_container_width=True):
                html = rep_gen.generate_html(ranked, jd, stats)
                st.download_button(
                    "⬇ Download HTML",
                    data=html,
                    file_name="hr_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.components.v1.html(html, height=700, scrolling=True)

        with col_json:
            st.markdown("### 📦 JSON Report")
            if st.button("Generate JSON Report", use_container_width=True):
                json_str = rep_gen.generate_json(ranked, jd, stats)
                st.download_button(
                    "⬇ Download JSON",
                    data=json_str,
                    file_name="hr_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.code(json_str[:4000] + ("..." if len(json_str) > 4000 else ""), language="json")
