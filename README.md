# Career Advisor

**GitHub repo:** [github.com/iamsubham019/carrer-advisor](https://github.com/iamsubham019/carrer-advisor)

An AI-powered tool that compares a resume against a job description and
produces a grounded fit score plus plain-language advice on what to fix and
how to rewrite it.

Built as a 5-agent pipeline (resume parser, JD parser, fit scorer, gap
analyzer, resume improver) orchestrated with LangGraph, with a single-file
frontend on top. Fully deployed and live.

## Live

- **Frontend:** https://subham-career-advisor-frontend.onrender.com
- **Backend API docs:** https://subham-career-advisor-backend.onrender.com/docs

Both run on Render's free tier, which spins down after inactivity — the
first request after idle time can take 30-60 seconds while the instance
wakes up. This is normal, not a bug.

## Status: fully built, tested, and deployed ✅

| Component | File | Status |
|---|---|---|
| Resume Parser Agent | `backend/agents/resume_parser.py` | ✅ tested with real Groq calls, live |
| JD Parser Agent | `backend/agents/jd_parser.py` | ✅ tested with real Groq calls, live |
| Fit Scorer | `backend/agents/fit_scorer.py` | ✅ tested with real embedding model, live |
| Gap Analysis Agent | `backend/agents/gap_analyzer.py` | ✅ tested with real Groq calls, live |
| Resume Improvement Agent | `backend/agents/resume_improver.py` | ✅ tested with real Groq calls, live |
| Orchestrator (LangGraph) | `backend/agents/orchestrator.py` | ✅ wires all 5 agents, gap+improve run in parallel |
| Frontend | `frontend/index.html` | ✅ single-file, no build step, deployed as static site |

Endpoints: `/parse-resume`, `/parse-jd`, `/score-fit`, `/analyze-gap`,
`/improve-resume` (individual, for testing/debugging), and `/analyze`
(single call, runs the full pipeline — what the frontend actually uses).

## Architecture notes

**Embeddings run on `fastembed` (ONNX Runtime), not `sentence-transformers`
(PyTorch).** The original PyTorch-based setup worked locally but crashed
Render's free-tier instance with an out-of-memory error (`torch` alone uses
300-700MB just loaded, exceeding the 512MB free-tier limit). Switched to
`fastembed` with `BAAI/bge-small-en-v1.5`, which does the same semantic
matching job on ONNX Runtime with a much smaller memory footprint. No
functional downside found in testing — matching quality held up.

**Python version pinned to 3.11.9 via a `PYTHON_VERSION` environment variable
on Render** (not `runtime.txt`, which Render's build system ignored despite
being the documented method). Render defaulted to Python 3.14, which doesn't
yet have prebuilt wheels for `pydantic-core`, causing a Rust-compilation
build failure. The environment variable approach worked reliably.

## How to run it locally

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

Then open `frontend/index.html` directly in your browser. Note: the deployed
`index.html` points `API_BASE` at the live Render backend URL, not
`localhost:8000` — change that line back if you want to test against a local
backend instead of the live one.

## Known limitations

**The Fit Scorer can miss connections between jargon and its plain-language
meaning**, though this improved noticeably after switching to
`BAAI/bge-small-en-v1.5`. It's still a general-purpose embedding model, good
at matching concepts phrased in similar language ("led a team" ≈ "leadership
experience") but capable of under-matching specific technical jargon that
isn't spelled out in plain language nearby. This is a deliberate design
tradeoff, not something to chase by tuning `MATCH_THRESHOLD` per-example --
that would loosen matching globally and produce more false positives across
every future resume/JD pair. The better fix lives in resume content: pairing
jargon with a plain-language explanation helps both this tool and real-world
ATS systems / human recruiters.

**The Resume Parser can silently drop terms that don't cleanly fit "skills"
vs "tools_and_frameworks."** Fixed once already (evaluation metrics like
F1-Score/ROC-AUC were being dropped entirely) by making the prompt explicit
that ambiguous terms should default into "skills" rather than being omitted.
Worth spot-checking the parsed output against the source resume periodically,
especially after resume format changes.

**Render free-tier cold starts.** Both services spin down after inactivity.
The frontend shows a status message warning about this during the wait, but
there's no way around the delay on the free tier without upgrading.

## Possible next steps (not built)

- Add caching so repeated runs on the same resume don't re-call the LLM
- Add a custom domain instead of the default `onrender.com` URLs
- Consider a lightweight ping/keep-alive to reduce cold-start frequency
  (tradeoff: uses free-tier hours faster)
