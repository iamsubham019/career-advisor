"""
Job Description Parser Agent.

Takes raw JD text (pasted or extracted from a file) -> returns a validated
ParsedJobDescription object. Same pattern as the Resume Parser Agent:
JSON-only system prompt + Pydantic validation + one retry on failure.

Deliberately reuses the same Groq client setup as resume_parser.py rather
than duplicating client init logic -- import it directly.
"""

import json
from pydantic import ValidationError

from schemas.jd_schema import ParsedJobDescription
from agents.resume_parser import _get_client, _strip_markdown_fences, MODEL

SYSTEM_PROMPT = """You are a precise job description parsing engine. Extract structured information from the job description text provided.

Return ONLY a valid JSON object with this exact structure, no preamble, no markdown fences, no commentary:

{
  "job_title": string or null,
  "company": string or null,
  "location": string or null,
  "employment_type": string or null,
  "experience_level": string or null,
  "must_have_skills": [list of strings -- skills explicitly stated as required/must-have],
  "nice_to_have_skills": [list of strings -- skills stated as preferred/bonus/nice-to-have],
  "tools_and_frameworks": [list of specific named tools, libraries, frameworks mentioned],
  "responsibilities": [list of strings -- key job duties/responsibilities],
  "qualifications": [list of strings -- required degree, certifications, years of experience],
  "summary": short 1-2 sentence summary of the role or null
}

Rules:
- Do not invent information that isn't in the text.
- If a field genuinely isn't present, use null (for scalars) or an empty list (for arrays).
- Distinguish must-have from nice-to-have based on the JD's own language (e.g. "required", "must have" vs "preferred", "a plus", "bonus"). If the JD doesn't distinguish, put everything under must_have_skills.
- Keep responsibilities and qualifications as separate concise bullet-style strings, not one giant paragraph."""


def _call_llm(jd_text: str, retry_hint: str = "") -> str:
    client = _get_client()
    user_content = f"Job description text:\n\n{jd_text}"
    if retry_hint:
        user_content += f"\n\n{retry_hint}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def parse_job_description(jd_text: str) -> ParsedJobDescription:
    """
    Main entry point: raw JD text -> validated ParsedJobDescription.
    Retries once on JSON/validation failure with corrective feedback to the LLM.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text is empty -- nothing to parse.")

    raw_output = _call_llm(jd_text)
    cleaned = _strip_markdown_fences(raw_output)

    try:
        data = json.loads(cleaned)
        parsed = ParsedJobDescription(**data)
        parsed.raw_text = jd_text
        return parsed
    except (json.JSONDecodeError, ValidationError) as first_error:
        retry_hint = (
            f"Your previous response was not valid JSON matching the schema. "
            f"Error: {str(first_error)[:300]}. "
            f"Return ONLY the corrected raw JSON object, nothing else."
        )
        raw_retry = _call_llm(jd_text, retry_hint=retry_hint)
        cleaned_retry = _strip_markdown_fences(raw_retry)

        try:
            data = json.loads(cleaned_retry)
            parsed = ParsedJobDescription(**data)
            parsed.raw_text = jd_text
            return parsed
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise RuntimeError(
                f"JD parsing failed after retry. "
                f"First error: {first_error}. Second error: {second_error}."
            )
