# Career Advisor AI

An AI-powered tool that compares a resume against a job description and
produces a grounded fit score plus plain-language advice on what to fix.

Built as a 5-agent pipeline (resume parser, JD parser, fit scorer, gap
analyzer, orchestrator) with a single-file frontend on top.

## Status: fully built and tested end-to-end ✅

| Component | File | Status |
|---|---|---|
| Resume Parser Agent | `backend/agents/resume_parser.py` | ✅ tested with real Groq calls |
| JD Parser Agent | `backend/agents/jd_parser.py` | ✅ tested with real Groq calls |
| Fit Scorer | `backend/agents/fit_scorer.py` | ✅ tested with real embedding model |
| Gap Analysis Agent | `backend/agents/gap_analyzer.py` | ✅ tested with real Groq calls |
| Orchestrator (LangGraph) | `backend/agents/orchestrator.py` | ✅ wires all 4 agents into one call |
| Frontend | `frontend/index.html` | ✅ single-file, no build step, tested in browser |

Endpoints: `/parse-resume`, `/parse-jd`, `/score-fit`, `/analyze-gap` (individual,
for testing/debugging), and `/analyze` (single call, runs the full pipeline —
what the frontend actually uses).

## How to run it

```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

Create `backend/.env` with:
```
GROQ_API_KEY=your_real_key_here
```

Start the server:
```bash
uvicorn main:app --reload
```

Then open `frontend/index.html` directly in your browser (no build step needed).
Backend must stay running at `http://localhost:8000` while you use it.

## Known limitations

**The Fit Scorer can miss connections between jargon and its plain-language
meaning.** It uses a general-purpose embedding model (`all-MiniLM-L6-v2`),
which is good at matching concepts phrased in similar language ("led a team"
≈ "leadership experience") but can under-match specific technical jargon that
isn't spelled out in plain language nearby. Example found during testing:
`GroupShuffleSplit` (a specific scikit-learn function) didn't strongly match
a JD's "preventing data leakage in train/test splits", even though that's
exactly what it does, because the resume didn't spell out the plain-language
meaning next to the jargon term.

**This is a deliberate design tradeoff, not a bug to chase with threshold
tuning.** Lowering `MATCH_THRESHOLD` to catch this one case would loosen
matching everywhere, producing more false positives across every future
resume/JD pair. The better fix lives in resume content: pairing jargon with
a plain-language explanation (e.g. "used GroupShuffleSplit to prevent
patient-level data leakage" instead of just "GroupShuffleSplit (zero patient
leakage)") helps both this tool and real-world ATS systems / human recruiters.

**The Resume Parser can silently drop terms that don't cleanly fit "skills"
vs "tools_and_frameworks."** Fixed once already (evaluation metrics like
F1-Score/ROC-AUC were being dropped entirely) by making the prompt explicit
that ambiguous terms should default into "skills" rather than being omitted.
Worth spot-checking the parsed output against the source resume periodically,
especially after resume format changes.

## Possible next steps (not built)

- Deploy backend (Render) + frontend (Vercel/static host), similar to the
  MedScribe AI setup
- Add a "resume improvement suggestions" branch to the LangGraph pipeline
  that runs in parallel with fit scoring
- Swap the embedding model for a domain-tuned one if jargon-matching accuracy
  becomes a real blocker rather than a documented edge case
- Add caching so repeated runs on the same resume don't re-call the LLM
