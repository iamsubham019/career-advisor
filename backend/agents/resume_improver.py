"""
Resume Improvement Agent.

Takes the parsed resume, parsed JD, and fit/gap results -> produces concrete
rewritten bullet text the candidate can paste directly into their resume.

Different job from the Gap Analysis Agent: gap_analysis.resume_wording_tips
gives abstract advice ("mention Python more"). This agent does the actual
rewrite -- takes a real existing bullet and returns an improved version that
surfaces the same true experience in language closer to what the JD (and
likely an ATS/recruiter scanning for these exact terms) is looking for.

Grounding rule, same as gap_analyzer: never invent experience the resume
doesn't already contain. A rewrite may reword or resurface an existing claim
in clearer language, but must not add skills, tools, or achievements that
aren't already present somewhere in the parsed resume.
"""

import json
from pydantic import ValidationError

from schemas.resume_schema import ParsedResume
from schemas.jd_schema import ParsedJobDescription
from schemas.fit_schema import FitScoreResult
from schemas.resume_improvement_schema import ResumeImprovementResult
from agents.resume_parser import _get_client, _strip_markdown_fences, MODEL

SYSTEM_PROMPT = """You are a resume editor helping a candidate tailor their resume to a specific job description.

You will be given the candidate's parsed resume (experience, projects, skills) and a job description, plus an already-computed fit score showing which required skills are matched vs missing.

Return ONLY a valid JSON object with this exact structure, no preamble, no markdown fences, no commentary:

{
  "rewritten_summary": "a 2-3 sentence professional summary tailored to this JD, using ONLY the candidate's real skills/experience -- or null if the existing summary is already strong",
  "bullet_rewrites": [
    {
      "section": "which resume section this bullet is from, e.g. 'Experience: IEEE EMBS Research Intern' or 'Project: SkinScan AI'",
      "original": "the original bullet/description text, verbatim from the input",
      "improved": "a rewritten version that surfaces the SAME underlying work but in language closer to the JD's terminology -- do not invent new claims",
      "reason": "one sentence on why this rewrite helps for THIS specific job"
    }
  ],
  "missing_skill_bullet_suggestions": [
    "for missing must-have skills the candidate might plausibly already have evidence for elsewhere but never wrote up as a bullet, a suggested NEW bullet idea framed as a suggestion, e.g. 'If you have used Git for version control, add a bullet like: ...' -- always frame these conditionally, never assert the candidate has unverified experience"
  ]
}

Rules:
- NEVER invent skills, tools, achievements, or experience that isn't already present somewhere in the candidate's parsed resume data.
- Only rewrite bullets where there's a real, specific improvement to make for this JD -- do not force rewrites on bullets that are already strong or irrelevant to this JD.
- For "missing_skill_bullet_suggestions", always phrase conditionally ("if you have experience with X...") since you cannot verify the candidate has unlisted experience -- never assert they do.
- Keep the same factual claims in each rewrite -- change wording/framing/keywords, not substance.
- If there's nothing meaningful to rewrite, return empty lists rather than forcing changes."""


def _call_llm(context_json: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input:\n\n{context_json}"},
        ],
        temperature=0.3,
        max_tokens=2500,
    )
    return response.choices[0].message.content.strip()


def improve_resume(
    resume: ParsedResume, jd: ParsedJobDescription, fit_result: FitScoreResult
) -> ResumeImprovementResult:
    """
    Main entry point: parsed resume + parsed JD + fit score -> ResumeImprovementResult.
    """
    context = {
        "resume": {
            "summary": resume.summary,
            "skills": resume.skills,
            "tools_and_frameworks": resume.tools_and_frameworks,
            "experience": [e.model_dump() for e in resume.experience],
            "projects": [p.model_dump() for p in resume.projects],
        },
        "job_description": {
            "job_title": jd.job_title,
            "must_have_skills": jd.must_have_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "responsibilities": jd.responsibilities,
        },
        "fit_score": {
            "overall_score": fit_result.overall_score,
            "missing_must_haves": fit_result.missing_must_haves,
        },
    }
    context_json = json.dumps(context, indent=2)

    raw_output = _call_llm(context_json)
    cleaned = _strip_markdown_fences(raw_output)

    try:
        data = json.loads(cleaned)
        return ResumeImprovementResult(**data)
    except (json.JSONDecodeError, ValidationError) as first_error:
        retry_prompt = context_json + (
            f"\n\nYour previous response was not valid JSON matching the schema. "
            f"Error: {str(first_error)[:300]}. Return ONLY the corrected raw JSON object."
        )
        raw_retry = _call_llm(retry_prompt)
        cleaned_retry = _strip_markdown_fences(raw_retry)

        try:
            data = json.loads(cleaned_retry)
            return ResumeImprovementResult(**data)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise RuntimeError(
                f"Resume improvement failed after retry. "
                f"First error: {first_error}. Second error: {second_error}."
            )
