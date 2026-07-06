"""
Gap Analysis Agent.

Takes the Fit Scorer's numeric output + the parsed resume + parsed JD ->
produces plain-language advice: what's strong, what's missing, and concrete
steps to close the gap. This is the one agent where LLM reasoning is the
right tool (explaining *why* something matters, in context), unlike the Fit
Scorer where a grounded number was the priority.

Grounding rule: the prompt explicitly hands the LLM the already-computed
match/no-match verdicts rather than asking it to re-decide fit from scratch.
This keeps the explanation consistent with the score instead of the LLM
contradicting a number it didn't compute.
"""

import json
from pydantic import ValidationError

from schemas.resume_schema import ParsedResume
from schemas.jd_schema import ParsedJobDescription
from schemas.fit_schema import FitScoreResult
from schemas.gap_analysis_schema import GapAnalysisResult
from agents.resume_parser import _get_client, _strip_markdown_fences, MODEL

SYSTEM_PROMPT = """You are a career advisor analyzing how well a candidate's resume fits a job description.

You will be given:
1. The candidate's parsed resume data
2. The job description's parsed requirements
3. An already-computed fit score with matched/missing skills (DO NOT recompute or contradict this -- treat it as ground truth)

Return ONLY a valid JSON object with this exact structure, no preamble, no markdown fences, no commentary:

{
  "overall_verdict": "1-2 sentence plain-language summary of how strong the fit is and why",
  "strengths": ["list of specific strengths -- reference actual matched skills/experience from the resume, explain why each is relevant to this specific role"],
  "key_gaps": ["list of specific gaps -- reference the actual missing must-have skills, explain why each matters for this role"],
  "actionable_suggestions": ["concrete next steps the candidate could take to close gaps -- e.g. a specific type of project, certification, or skill to develop"],
  "resume_wording_tips": ["specific phrasing suggestions -- e.g. if the candidate has relevant experience but didn't phrase it in the JD's terms, suggest how to reword it"]
}

Rules:
- Base everything on the actual data provided -- do not invent skills, experience, or requirements not present in the input.
- Be specific and reference real details (project names, skill names) rather than generic advice.
- If the fit score's missing_must_haves list is empty, say so honestly in key_gaps (e.g. "No major must-have gaps identified") rather than inventing a gap.
- Keep each list item concise (1-2 sentences) -- this will be displayed as a bulleted list, not paragraphs."""


def _call_llm(context_json: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analysis input:\n\n{context_json}"},
        ],
        temperature=0.3,  # slightly higher than the parsers -- this is advisory writing, not extraction
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def analyze_gap(
    resume: ParsedResume, jd: ParsedJobDescription, fit_result: FitScoreResult
) -> GapAnalysisResult:
    """
    Main entry point: parsed resume + parsed JD + fit score -> GapAnalysisResult.
    """
    context = {
        "resume_summary": {
            "skills": resume.skills,
            "tools_and_frameworks": resume.tools_and_frameworks,
            "experience": [e.model_dump() for e in resume.experience],
            "projects": [p.model_dump() for p in resume.projects],
        },
        "job_description_summary": {
            "job_title": jd.job_title,
            "must_have_skills": jd.must_have_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "responsibilities": jd.responsibilities,
            "qualifications": jd.qualifications,
        },
        "computed_fit_score": {
            "overall_score": fit_result.overall_score,
            "missing_must_haves": fit_result.missing_must_haves,
            "extra_resume_skills": fit_result.extra_resume_skills,
            "matched_skill_count": fit_result.matched_skill_count,
            "total_required_skill_count": fit_result.total_required_skill_count,
        },
    }
    context_json = json.dumps(context, indent=2)

    raw_output = _call_llm(context_json)
    cleaned = _strip_markdown_fences(raw_output)

    try:
        data = json.loads(cleaned)
        return GapAnalysisResult(**data)
    except (json.JSONDecodeError, ValidationError) as first_error:
        retry_prompt = context_json + (
            f"\n\nYour previous response was not valid JSON matching the schema. "
            f"Error: {str(first_error)[:300]}. Return ONLY the corrected raw JSON object."
        )
        raw_retry = _call_llm(retry_prompt)
        cleaned_retry = _strip_markdown_fences(raw_retry)

        try:
            data = json.loads(cleaned_retry)
            return GapAnalysisResult(**data)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise RuntimeError(
                f"Gap analysis failed after retry. "
                f"First error: {first_error}. Second error: {second_error}."
            )
