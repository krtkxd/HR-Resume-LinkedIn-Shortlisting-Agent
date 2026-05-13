"""
report/generator.py
────────────────────
Report Generator

Produces:
  1. JSON output (raw dict / .json file)
  2. Styled HTML report (Jinja2 template)
  3. Optional PDF (ReportLab)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# HTML Template (embedded, no external files)
# ──────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HR Shortlisting Report – {{ job_title }}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 1300px; margin: 0 auto; padding: 40px 24px; }

  /* Header */
  .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 36px 40px; margin-bottom: 32px; }
  .header h1 { font-size: 2rem; font-weight: 700; color: #fff; }
  .header p { color: rgba(255,255,255,0.75); margin-top: 6px; font-size: 0.95rem; }
  .meta-grid { display: flex; gap: 32px; margin-top: 20px; flex-wrap: wrap; }
  .meta-item { background: rgba(255,255,255,0.15); border-radius: 10px;
    padding: 12px 20px; backdrop-filter: blur(8px); }
  .meta-item .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(255,255,255,0.6); }
  .meta-item .value { font-size: 1.1rem; font-weight: 600; color: #fff; margin-top: 2px; }

  /* Stats bar */
  .stats-bar { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px; margin-bottom: 32px; }
  .stat-card { background: #1e2030; border: 1px solid #2d3148;
    border-radius: 12px; padding: 20px; text-align: center; }
  .stat-card .num { font-size: 2rem; font-weight: 700; color: #818cf8; }
  .stat-card .desc { font-size: 0.8rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

  /* Table */
  .table-wrapper { background: #1e2030; border: 1px solid #2d3148; border-radius: 16px; overflow: hidden; margin-bottom: 32px; }
  table { width: 100%; border-collapse: collapse; }
  thead { background: #252840; }
  th { padding: 14px 16px; text-align: left; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94a3b8; font-weight: 600; white-space: nowrap; }
  td { padding: 14px 16px; border-top: 1px solid #252840; font-size: 0.875rem; vertical-align: middle; }
  tr:hover td { background: #252840; transition: background 0.15s; }

  /* Rank badge */
  .rank-badge { display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 50%; font-weight: 700; font-size: 0.85rem; }
  .rank-1 { background: linear-gradient(135deg,#f59e0b,#d97706); color:#fff; }
  .rank-2 { background: linear-gradient(135deg,#94a3b8,#64748b); color:#fff; }
  .rank-3 { background: linear-gradient(135deg,#b45309,#92400e); color:#fff; }
  .rank-n { background: #252840; color: #94a3b8; }

  /* Score bar */
  .score-bar-wrap { display: flex; align-items: center; gap: 10px; }
  .score-bar { flex: 1; height: 6px; background: #2d3148; border-radius: 3px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
  .score-text { font-weight: 600; min-width: 28px; text-align: right; }

  /* Mini scores */
  .mini-scores { display: flex; gap: 6px; flex-wrap: wrap; }
  .mini-score { padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 600;
    background: #252840; color: #94a3b8; white-space: nowrap; }
  .mini-score.high { background: #064e3b; color: #34d399; }
  .mini-score.mid  { background: #1e3a5f; color: #60a5fa; }
  .mini-score.low  { background: #4c1d1d; color: #f87171; }

  /* Recommendation pill */
  .pill { display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
  .pill-strong-hire { background:#065f46; color:#34d399; }
  .pill-hire        { background:#1e3a5f; color:#60a5fa; }
  .pill-maybe       { background:#713f12; color:#fbbf24; }
  .pill-no-hire     { background:#4c1d1d; color:#f87171; }

  /* Tier badge */
  .tier { display: inline-block; width: 24px; height: 24px; line-height: 24px;
    text-align: center; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
  .tier-A { background:#065f46; color:#34d399; }
  .tier-B { background:#1e3a5f; color:#60a5fa; }
  .tier-C { background:#713f12; color:#fbbf24; }
  .tier-D { background:#4c1d1d; color:#f87171; }

  /* Detail sections */
  .detail-section { margin-top: 32px; }
  .detail-section h2 { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; margin-bottom: 16px;
    padding-bottom: 8px; border-bottom: 1px solid #2d3148; }
  .candidate-card { background: #1e2030; border: 1px solid #2d3148; border-radius: 12px;
    padding: 24px; margin-bottom: 16px; }
  .candidate-card h3 { font-size: 1rem; font-weight: 600; color: #c7d2fe; margin-bottom: 4px; }
  .candidate-card .sub { font-size: 0.8rem; color: #64748b; margin-bottom: 16px; }
  .rubric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .rubric-item { background: #252840; border-radius: 8px; padding: 14px; }
  .rubric-item .cat { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #64748b; margin-bottom: 6px; }
  .rubric-item .sc { font-size: 1.4rem; font-weight: 700; color: #818cf8; }
  .rubric-item .just { font-size: 0.75rem; color: #94a3b8; margin-top: 6px; line-height: 1.4; }
  .skill-gap { margin-top: 12px; }
  .skill-gap .label { font-size: 0.75rem; color: #f87171; margin-bottom: 6px; }
  .gap-tag { display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 4px;
    background: #4c1d1d; color: #fca5a5; font-size: 0.72rem; }

  /* Footer */
  footer { text-align: center; color: #374151; font-size: 0.8rem; padding-top: 32px; }
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>🎯 HR Shortlisting Report</h1>
    <p>AI-Powered Candidate Evaluation Pipeline</p>
    <div class="meta-grid">
      <div class="meta-item"><div class="label">Position</div><div class="value">{{ job_title }}</div></div>
      <div class="meta-item"><div class="label">Generated</div><div class="value">{{ generated_at }}</div></div>
      <div class="meta-item"><div class="label">Candidates</div><div class="value">{{ stats.total_candidates }}</div></div>
      <div class="meta-item"><div class="label">Recommended Hires</div><div class="value">{{ stats.hire_count }}</div></div>
    </div>
  </div>

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat-card"><div class="num">{{ stats.total_candidates }}</div><div class="desc">Total Evaluated</div></div>
    <div class="stat-card"><div class="num">{{ stats.hire_count }}</div><div class="desc">Hire / Strong Hire</div></div>
    <div class="stat-card"><div class="num">{{ "%.1f"|format(stats.average_score) }}</div><div class="desc">Avg Score</div></div>
    <div class="stat-card"><div class="num">{{ "%.1f"|format(stats.highest_score) }}</div><div class="desc">Top Score</div></div>
    {% for tier, count in stats.tier_distribution.items() %}
    <div class="stat-card"><div class="num">{{ count }}</div><div class="desc">Tier {{ tier }}</div></div>
    {% endfor %}
  </div>

  <!-- Ranked table -->
  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Candidate</th>
          <th>Tier</th>
          <th>Skills (30%)</th>
          <th>Experience (25%)</th>
          <th>Education (15%)</th>
          <th>Projects (20%)</th>
          <th>Communication (10%)</th>
          <th>Total Score</th>
          <th>Recommendation</th>
        </tr>
      </thead>
      <tbody>
      {% for c in candidates %}
      <tr>
        <td>
          <span class="rank-badge {% if c.rank == 1 %}rank-1{% elif c.rank == 2 %}rank-2{% elif c.rank == 3 %}rank-3{% else %}rank-n{% endif %}">
            {{ c.rank }}
          </span>
        </td>
        <td>
          <div style="font-weight:600;color:#e2e8f0;">{{ c.name }}</div>
          <div style="font-size:0.75rem;color:#64748b;">{{ c.source | upper }} · {{ c.percentile }}th pct</div>
        </td>
        <td><span class="tier tier-{{ c.tier }}">{{ c.tier }}</span></td>
        {% for cat in ['skills_match','experience','education','projects','communication'] %}
        {% set s = c.scores[cat].score %}
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar">
              <div class="score-fill" style="width:{{ s*10 }}%;background:{% if s>=8 %}#34d399{% elif s>=5 %}#60a5fa{% else %}#f87171{% endif %};"></div>
            </div>
            <span class="score-text" style="color:{% if s>=8 %}#34d399{% elif s>=5 %}#60a5fa{% else %}#f87171{% endif %}">{{ s }}</span>
          </div>
        </td>
        {% endfor %}
        <td>
          <div style="font-size:1.3rem;font-weight:700;color:{% if c.total_score>=8 %}#34d399{% elif c.total_score>=6.5 %}#60a5fa{% elif c.total_score>=5 %}#fbbf24{% else %}#f87171{% endif %}">
            {{ "%.1f"|format(c.total_score) }}
          </div>
        </td>
        <td>
          <span class="pill pill-{{ c.recommendation | lower | replace(' ','-') }}">
            {{ c.recommendation }}
          </span>
        </td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Detailed breakdown -->
  <div class="detail-section">
    <h2>📋 Detailed Candidate Breakdown</h2>
    {% for c in candidates %}
    <div class="candidate-card">
      <h3>{{ c.rank }}. {{ c.name }}</h3>
      <div class="sub">
        Score: {{ "%.2f"|format(c.total_score) }}/10 · {{ c.recommendation }} · Tier {{ c.tier }} · Confidence: {{ (c.confidence*100)|int }}%
      </div>
      <div class="rubric-grid">
        {% for cat, data in c.scores.items() %}
        <div class="rubric-item">
          <div class="cat">{{ cat | replace('_',' ') | title }} ({{ (data.weight*100)|int }}%)</div>
          <div class="sc">{{ data.score }}/10</div>
          <div class="just">{{ data.justification }}</div>
        </div>
        {% endfor %}
      </div>
      {% if c.skill_gaps %}
      <div class="skill-gap">
        <div class="label">⚠ Missing Skills:</div>
        {% for gap in c.skill_gaps[:10] %}
        <span class="gap-tag">{{ gap }}</span>
        {% endfor %}
      </div>
      {% endif %}
      {% if c.override_log %}
      <div style="margin-top:12px;padding:10px;background:#1a1a2e;border-radius:6px;font-size:0.75rem;color:#94a3b8;">
        <strong style="color:#fbbf24;">Manual Overrides:</strong>
        {% for ov in c.override_log %}
        <div>{{ ov.timestamp }} — {{ ov.category }}: {{ ov.old_score }}→{{ ov.new_score }} by {{ ov.overrider }} ({{ ov.reason }})</div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </div>

  <footer>
    Generated by HR Resume & LinkedIn Shortlisting Agent · {{ generated_at }}
  </footer>
</div>
</body>
</html>"""


# ──────────────────────────────────────────────
# Generator class
# ──────────────────────────────────────────────

class ReportGenerator:
    """
    Generates reports in JSON and HTML formats.
    Optional PDF generation via ReportLab.
    """

    def generate_json(
        self,
        ranked_candidates: List[Dict[str, Any]],
        jd: Dict[str, Any],
        stats: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Serialize results to JSON string (and optionally save to file).
        """
        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "job_title": jd.get("title", "Unknown Position"),
            "jd_summary": {
                "required_skills": jd.get("required_skills", [])[:10],
                "experience_level": jd.get("experience_level", ""),
                "domain": jd.get("domain", ""),
                "education_required": jd.get("education", ""),
            },
            "statistics": stats,
            "candidates": ranked_candidates,
        }
        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def generate_html(
        self,
        ranked_candidates: List[Dict[str, Any]],
        jd: Dict[str, Any],
        stats: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render the Jinja2 HTML template and return HTML string.
        """
        try:
            from jinja2 import Template
            template = Template(_HTML_TEMPLATE)
        except ImportError:
            # Fallback: simple string interpolation
            return self._simple_html(ranked_candidates, jd, stats)

        html = template.render(
            candidates=ranked_candidates,
            jd=jd,
            stats=stats,
            job_title=jd.get("title", "Unknown Position"),
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def generate_pdf(
        self,
        ranked_candidates: List[Dict[str, Any]],
        jd: Dict[str, Any],
        stats: Dict[str, Any],
        output_path: str = "report.pdf",
    ) -> str:
        """Generate a PDF report using ReportLab."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            )
        except ImportError:
            raise ImportError(
                "reportlab is not installed. Run: pip install reportlab"
            )

        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(A4),
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph(
            f"<b>HR Shortlisting Report – {jd.get('title','')}</b>",
            styles["Title"]
        ))
        story.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Total Candidates: {stats.get('total_candidates',0)} | "
            f"Recommended: {stats.get('hire_count',0)}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5 * cm))

        # Table
        header = [
            "Rank", "Name", "Skills", "Exp", "Edu", "Projects", "Comm",
            "Total", "Recommendation"
        ]
        table_data = [header]
        for c in ranked_candidates:
            sc = c["scores"]
            table_data.append([
                str(c["rank"]),
                c["name"],
                str(sc["skills_match"]["score"]),
                str(sc["experience"]["score"]),
                str(sc["education"]["score"]),
                str(sc["projects"]["score"]),
                str(sc["communication"]["score"]),
                f"{c['total_score']:.1f}",
                c["recommendation"],
            ])

        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, -1), 20),
        ]))
        story.append(tbl)
        doc.build(story)
        return output_path

    def _simple_html(
        self,
        ranked_candidates: List[Dict[str, Any]],
        jd: Dict[str, Any],
        stats: Dict[str, Any],
    ) -> str:
        """Minimal fallback HTML without Jinja2."""
        rows = ""
        for c in ranked_candidates:
            sc = c["scores"]
            rows += f"""<tr>
              <td>{c['rank']}</td><td>{c['name']}</td>
              <td>{sc['skills_match']['score']}</td>
              <td>{sc['experience']['score']}</td>
              <td>{sc['education']['score']}</td>
              <td>{sc['projects']['score']}</td>
              <td>{sc['communication']['score']}</td>
              <td><b>{c['total_score']}</b></td>
              <td>{c['recommendation']}</td>
            </tr>"""
        return f"""<!DOCTYPE html><html><head><title>HR Report</title></head>
<body><h1>HR Shortlisting Report – {jd.get('title','')}</h1>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Rank</th><th>Name</th><th>Skills</th><th>Exp</th>
<th>Edu</th><th>Projects</th><th>Comm</th><th>Total</th><th>Recommendation</th></tr>
{rows}</table></body></html>"""
