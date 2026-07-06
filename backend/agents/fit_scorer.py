"""
Fit Scorer Agent.

Computes a grounded, numeric fit score between a parsed resume and a parsed
job description using sentence embeddings + cosine similarity -- NOT keyword
matching, so "led a team of 5 engineers" can match "leadership experience"
and "PyTorch" can match "deep learning framework experience".

Why this exists as its own non-LLM step (same principle as SkinScan's OOD
threshold): the Gap Analysis Agent (LLM) is good at *explaining* a fit, but
LLMs are unreliable at producing a consistent, reproducible numeric score on
their own. Compute the number here, deterministically; let the LLM reason
about it afterward.

Important: the matching pool is NOT just resume.skills + tools_and_frameworks.
Those two lists are often short and incomplete -- someone's actual evidence of
a skill (e.g. "prevented data leakage using GroupShuffleSplit") usually lives
in project/experience descriptions, not in a one-word skills list. So we also
pull short claims out of project and experience descriptions and add them as
separate candidates. Without this, the scorer only sees what was explicitly
typed as a "skill", and misses real, demonstrated experience -- which is
exactly the kind of false-negative gap analysis shouldn't produce.

Model note: first run downloads 'all-MiniLM-L6-v2' (~80MB) from Hugging Face,
so it needs an internet connection the first time. After that it's cached
locally and loads fast.
"""

import re
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from schemas.resume_schema import ParsedResume
from schemas.jd_schema import ParsedJobDescription
from schemas.fit_schema import FitScoreResult, SkillMatch

MATCH_THRESHOLD = 0.45  # similarity above this counts as "matched"; tuned loosely, adjust after real testing
MAX_DISPLAY_LEN = 90  # truncate long matched candidate text (e.g. a whole project sentence) for readability

_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load so importing this module doesn't trigger a download until actually needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _split_into_clauses(text: str) -> List[str]:
    """
    Break a longer description into short, individual claims so each one can
    be embedded and matched independently.

    Resume bullet points often pack several distinct technical claims into
    one sentence, e.g.:
      "integrated Grad-CAM explainability, GroupShuffleSplit (zero patient
       leakage), and 45% OOD rejection"
    Splitting only on periods/semicolons leaves all three claims merged into
    one candidate, which dilutes the embedding so badly that none of them
    matches their JD counterpart well. Splitting on commas and standalone
    "and" as well breaks this into three independent candidates:
      "integrated Grad-CAM explainability"
      "GroupShuffleSplit (zero patient leakage)"
      "45% OOD rejection"
    each of which can now be matched on its own merits.
    """
    if not text:
        return []
    # split on sentence terminators, bullet/separator characters, commas, and standalone "and" --
    # but NOT on a period that's part of a decimal number (e.g. "99.21%", "AUC 0.9997"),
    # using lookbehind/lookahead to require a period is not flanked by digits on both sides
    parts = re.split(r"(?<!\d)\.(?!\d)|[;\n•◦·]|,\s*|\band\b", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 4]


def _build_candidate_pool(resume: ParsedResume) -> List[str]:
    """
    Build the full set of short text candidates the resume offers as
    evidence, beyond just the bare skills/tools lists.
    """
    candidates: List[str] = list(resume.skills) + list(resume.tools_and_frameworks)

    for project in resume.projects:
        candidates.extend(project.tech_stack)
        candidates.extend(_split_into_clauses(project.description or ""))

    for exp in resume.experience:
        candidates.extend(exp.skills_used)
        candidates.extend(_split_into_clauses(exp.description or ""))

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        key = c.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _truncate(text: str, max_len: int = MAX_DISPLAY_LEN) -> str:
    if text and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _best_match_for_each(
    targets: List[str], candidates: List[str]
) -> List[Tuple[str, str, float]]:
    """
    For each item in `targets` (e.g. JD required skills), find the single best-matching
    item in `candidates` (e.g. resume skills+tools+project/experience clauses) by cosine similarity.
    Returns list of (target, best_candidate_or_None, similarity_score).
    """
    if not targets:
        return []
    if not candidates:
        return [(t, None, 0.0) for t in targets]

    model = _get_model()
    target_embeddings = model.encode(targets)
    candidate_embeddings = model.encode(candidates)

    sim_matrix = cosine_similarity(target_embeddings, candidate_embeddings)

    results = []
    for i, target in enumerate(targets):
        best_idx = sim_matrix[i].argmax()
        best_score = float(sim_matrix[i][best_idx])
        best_candidate = candidates[best_idx]
        results.append((target, best_candidate, best_score))
    return results


def score_fit(resume: ParsedResume, jd: ParsedJobDescription) -> FitScoreResult:
    """
    Main entry point: parsed resume + parsed JD -> FitScoreResult.

    Weighting: must-have skills count 2x toward the overall score vs nice-to-haves,
    since failing a must-have matters more than missing a bonus skill.
    """
    resume_pool = _build_candidate_pool(resume)
    # Keep a separate, smaller pool of just skills/tools for the "extra skills" report --
    # showing a raw project sentence there would look odd, extra_resume_skills should
    # only ever be actual named skills/tools.
    named_skills_pool = list(resume.skills) + list(resume.tools_and_frameworks)

    must_have_raw = _best_match_for_each(jd.must_have_skills, resume_pool)
    nice_to_have_raw = _best_match_for_each(jd.nice_to_have_skills, resume_pool)

    must_have_matches = [
        SkillMatch(
            jd_skill=t,
            matched_resume_item=_truncate(c),
            similarity=round(s, 3),
            is_matched=s >= MATCH_THRESHOLD,
        )
        for t, c, s in must_have_raw
    ]
    nice_to_have_matches = [
        SkillMatch(
            jd_skill=t,
            matched_resume_item=_truncate(c),
            similarity=round(s, 3),
            is_matched=s >= MATCH_THRESHOLD,
        )
        for t, c, s in nice_to_have_raw
    ]

    # Weighted score: must-haves worth 2 points each, nice-to-haves worth 1 point each
    must_have_weight = 2
    nice_to_have_weight = 1

    total_possible = (len(must_have_matches) * must_have_weight) + (
        len(nice_to_have_matches) * nice_to_have_weight
    )
    earned = (
        sum(must_have_weight for m in must_have_matches if m.is_matched)
        + sum(nice_to_have_weight for m in nice_to_have_matches if m.is_matched)
    )

    overall_score = round((earned / total_possible) * 100, 1) if total_possible > 0 else 0.0

    matched_count = sum(1 for m in must_have_matches + nice_to_have_matches if m.is_matched)
    total_required = len(must_have_matches) + len(nice_to_have_matches)

    missing_must_haves = [m.jd_skill for m in must_have_matches if not m.is_matched]

    matched_named_skills = set()
    for m in must_have_matches + nice_to_have_matches:
        if m.is_matched and m.matched_resume_item in named_skills_pool:
            matched_named_skills.add(m.matched_resume_item)
    extra_resume_skills = [s for s in named_skills_pool if s not in matched_named_skills]

    return FitScoreResult(
        overall_score=overall_score,
        must_have_matches=must_have_matches,
        nice_to_have_matches=nice_to_have_matches,
        matched_skill_count=matched_count,
        total_required_skill_count=total_required,
        missing_must_haves=missing_must_haves,
        extra_resume_skills=extra_resume_skills,
    )
