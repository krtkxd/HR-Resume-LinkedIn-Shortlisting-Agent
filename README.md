# 🎯 HR Resume & LinkedIn Shortlisting Agent

A **production-ready AI agent pipeline** that evaluates candidates against a Job Description (JD) and produces a ranked shortlist with transparent, explainable scoring.

---

## 📌 Problem Statement

Recruiters face an overwhelming volume of resumes. Manual review is slow, inconsistent, and prone to bias. This system automates the initial screening by:

1. Parsing job descriptions into structured requirements
2. Extracting candidate profiles from resumes (PDF/DOCX) and LinkedIn data
3. Scoring each candidate across 5 weighted rubric dimensions using semantic AI + rule-based logic
4. Ranking candidates with full score breakdowns and justifications
5. Allowing HR to apply manual overrides with audit trails

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Input Layer                                    │
│   JD Text │ Resume Files (PDF/DOCX) │ LinkedIn Profile (text/JSON)   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │           JD Parser Agent               │
         │  LLM → structured JSON (skills,         │
         │  experience, education, domain)         │
         └───────────────────┬────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │         Profile Parser Agent            │
         │  LLM + regex → candidate profiles      │
         │  (skills, exp, edu, projects)           │
         └───────────────────┬────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │      Semantic Matching Engine            │
         │  SentenceTransformers / OpenAI Embed    │
         │  Cosine similarity + Skill overlap      │
         └───────────────────┬────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │           Scoring Agent                  │
         │  5-category rubric × weighted scores    │
         │  + justification per category           │
         └───────────────────┬────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │           Ranking Agent                  │
         │  Sort by total score, assign tier       │
         └───────────────────┬────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │         Report Generator                 │
         │  JSON │ Styled HTML │ Optional PDF       │
         └────────────────────────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │       Human-in-the-Loop Override        │
         │  Manual score override + audit log      │
         └────────────────────────────────────────┘
```

---

## 📊 Scoring Rubric

| Category | Weight | Low (0–3) | Mid (4–7) | High (8–10) |
|---|---|---|---|---|
| **Skills Match** | 30% | <30% skill overlap | 50–70% | >85% |
| **Experience Relevance** | 25% | Unrelated domain | Adjacent | Exact domain & seniority |
| **Education & Certs** | 15% | Below minimum | Meets minimum | Exceeds + certs |
| **Projects / Portfolio** | 20% | None | Generic | Strong & relevant |
| **Communication Quality** | 10% | Poor structure | Average | Well-structured |

Scores combine **semantic embedding similarity** (60%) + **lexical skill overlap** (40%).

---

## 🗂️ Project Structure

```
hr_shortlisting_agent/
├── agents/
│   ├── jd_parser.py          # JD → structured JSON (LLM + regex fallback)
│   ├── profile_parser.py     # Resume/LinkedIn → candidate profile
│   ├── scoring_agent.py      # 5-category rubric scoring + overrides
│   └── ranking_agent.py      # Sort + tier assignment
│
├── core/
│   ├── embedding.py          # SentenceTransformers / OpenAI / Gemini
│   └── similarity.py         # Cosine, Jaccard, hybrid similarity
│
├── utils/
│   ├── resume_parser.py      # PDF (pdfplumber/PyMuPDF) + DOCX extraction
│   └── linkedin_parser.py    # LinkedIn JSON export + plain text parser
│
├── api/
│   └── main.py               # FastAPI: /analyze, /override, /results
│
├── report/
│   └── generator.py          # JSON + styled HTML + optional PDF (ReportLab)
│
├── ui/
│   └── streamlit_app.py      # Interactive Streamlit UI
│
├── data/
│   └── skills.json           # Skill taxonomy (500+ skills across categories)
│
├── samples/
│   ├── sample_jd.txt         # Sample ML Engineer JD
│   ├── resume_aisha_patel.txt    # Strong candidate (7 yrs ML)
│   ├── resume_marcus_johnson.txt # Mid-level candidate (3 yrs dev)
│   └── resume_priya_sharma.txt   # Research-focused candidate (5 yrs)
│
├── demo.py                   # End-to-end demo script
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Clone & Install

```bash
cd hr_shortlisting_agent
pip install -r requirements.txt
```

> **Minimum install** (offline mode, no API keys needed):
> ```bash
> pip install sentence-transformers pdfplumber python-docx fastapi uvicorn streamlit jinja2 pydantic
> ```

### 3. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env and add your API keys (OpenAI / Gemini / Anthropic)
```

The system works fully **without any API keys** — it uses SentenceTransformers (offline) for embeddings and regex-based parsing as fallback.

---

## 🚀 Running the System

### Option A: Interactive Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Opens at `http://localhost:8501`

### Option B: FastAPI Backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option C: Command-line Demo

```bash
python demo.py
```

Reads sample JD + resumes, generates `outputs/report.html` and `outputs/report.json`.

---

## 🌐 API Reference

### `POST /analyze`

Analyze candidates against a JD.

**Form fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `jd_text` | string | ✅ | Job description text |
| `resumes` | files | ⚠️ | PDF/DOCX resume files (multiple) |
| `linkedin_text` | string | ➖ | LinkedIn profile text or JSON |

**Response:**
```json
{
  "session_id": "uuid",
  "job_title": "Senior ML Engineer",
  "candidates": [...],
  "statistics": {
    "total_candidates": 3,
    "average_score": 6.82,
    "highest_score": 8.45,
    "hire_count": 2,
    "tier_distribution": {"A": 1, "B": 1, "C": 1}
  }
}
```

---

### `POST /override`

Apply a manual score override.

```json
{
  "session_id": "uuid",
  "candidate_id": "abc12345",
  "category": "projects",
  "new_score": 8,
  "reason": "Reviewed GitHub portfolio — strong ML projects not listed on resume",
  "overrider": "Jane HR Manager"
}
```

---

### `GET /results/{session_id}?format=html`

Retrieve results as JSON (default) or rendered HTML report.

---

## 📤 Example Output

```json
{
  "name": "Aisha Patel",
  "scores": {
    "skills_match": {
      "score": 9,
      "weight": 0.3,
      "justification": "Matched 18/22 required skills: python, pytorch, transformers, mlflow..."
    },
    "experience": {
      "score": 8,
      "weight": 0.25,
      "justification": "7 yrs experience; JD requires 5–10 yrs. Domain relevance: 84%."
    },
    "education": {
      "score": 8,
      "weight": 0.15,
      "justification": "Candidate: master (rank 4); JD requires: bachelor (rank 3). Certifications: 3."
    },
    "projects": {
      "score": 9,
      "weight": 0.2,
      "justification": "3 project(s) found. Relevance to JD domain 'Machine Learning / AI': 87%."
    },
    "communication": {
      "score": 8,
      "weight": 0.1,
      "justification": "Excellent communication. 412 words, 6 structured sections detected."
    }
  },
  "total_score": 8.65,
  "recommendation": "Strong Hire",
  "skill_gaps": ["horovod", "deepspeed"],
  "confidence": 1.0
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM Parsing | OpenAI GPT-4o-mini / Gemini 1.5 Flash / Claude 3 Haiku |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` (local) or OpenAI/Gemini |
| PDF Parsing | pdfplumber (primary) + PyMuPDF (fallback) |
| DOCX Parsing | python-docx |
| Backend API | FastAPI + Uvicorn |
| UI | Streamlit |
| HTML Reports | Jinja2 |
| PDF Reports | ReportLab |
| Validation | Pydantic v2 |

---

## 🔮 Future Improvements

1. **Bias Detection** — Analyse score distributions across demographic signals; flag potential disparate impact
2. **Interview Question Generator** — Auto-generate role-specific interview questions based on skill gaps
3. **ATS Integration** — Connect to Workday, Greenhouse, Lever via APIs
4. **Database persistence** — Replace in-memory session store with PostgreSQL
5. **Async processing** — Queue large batches via Celery + Redis
6. **Multi-language support** — Parse non-English resumes
7. **Feedback loop** — Learn from historical hire/no-hire decisions to refine weights
8. **Confidence calibration** — Better confidence estimates from parse completeness

---

## 📋 Notes on Fairness

This system evaluates candidates on **skills, experience, education, and portfolio quality only**. Names, gender, ethnicity, and other demographic attributes are never used as scoring inputs. HR teams are encouraged to:

- Use skill gaps as coaching opportunities, not disqualifiers
- Apply overrides when context is missing from the resume
- Maintain diverse interview panels for final decisions

---

## 📄 License

MIT License — Free for commercial and personal use.
