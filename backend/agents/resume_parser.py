"""
Resume Parser Agent.

Takes raw resume text -> returns a validated ParsedResume object.
Uses Groq LLaMA 3.3 70B (same backbone as MedScribe AI) with a strict
JSON-only system prompt, then validates the output through Pydantic.

If the LLM returns malformed JSON, we retry once with an explicit
"fix your JSON" nudge before giving up -- don't silently return garbage.
"""

import os
import json
from groq import Groq
from pydantic import ValidationError

from schemas.resume_schema import ParsedResume

MODEL = "llama-3.3-70b-versatile"

_client = None


def _get_client() -> Groq:
    """Lazy-init so importing this module doesn't crash when GROQ_API_KEY isn't set yet."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _client = Groq(api_key=api_key)
    return _client

SYSTEM_PROMPT = """You are a precise resume parsing engine. Extract structured information from the resume text provided.

Return ONLY a valid JSON object with this exact structure, no preamble, no markdown fences, no commentary:

{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "links": [list of URLs - github, linkedin, portfolio],
  "summary": short professional summary string or null,
  "skills": [list of technical/soft skill strings],
  "tools_and_frameworks": [list of specific tools, libraries, frameworks],
  "education": [{"degree": str, "institution": str, "field_of_study": str or null, "start_year": str or null, "end_year": str or null, "cgpa_or_grade": str or null}],
  "experience": [{"title": str, "organization": str, "duration": str or null, "description": str or null, "skills_used": [list of strings]}],
  "projects": [{"name": str, "description": str or null, "tech_stack": [list of strings]}],
  "certifications": [list of certification name strings]
}

Rules:
- Do not invent information that isn't in the text.
- If a field genuinely isn't present, use null (for scalars) or an empty list (for arrays).
- Separate "skills" (general competencies like "Leadership", "Deep Learning") from "tools_and_frameworks" (specific named tools like "PyTorch", "FastAPI").
- CRITICAL: every specific technical term, metric, methodology, or named technique listed anywhere in the resume (e.g. under a "Technical Skills" or "XAI & Evaluation" heading) MUST end up somewhere in the output -- never silently drop a term just because it doesn't obviously fit "skills" or "tools_and_frameworks". Evaluation metrics (F1-Score, ROC-AUC, Confusion Matrix), techniques (GroupShuffleSplit, Test-Time Augmentation), and methodologies all count as skills if there's no better bucket -- when in doubt, put the term in "skills" rather than omitting it.
- Keep descriptions concise -- summarize, don't copy paragraphs verbatim."""


def _call_llm(resume_text: str, retry_hint: str = "") -> str:
    user_content = f"Resume text:\n\n{resume_text}"
    if retry_hint:
        user_content += f"\n\n{retry_hint}"

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,  # low temp -- this is extraction, not creative writing
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def _strip_markdown_fences(raw: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions. Strip it defensively."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def parse_resume(resume_text: str) -> ParsedResume:
    """
    Main entry point: raw resume text -> validated ParsedResume.
    Retries once on JSON/validation failure with corrective feedback to the LLM.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text is empty -- nothing to parse.")

    raw_output = _call_llm(resume_text)
    cleaned = _strip_markdown_fences(raw_output)

    try:
        data = json.loads(cleaned)
        parsed = ParsedResume(**data)
        parsed.raw_text = resume_text
        return parsed
    except (json.JSONDecodeError, ValidationError) as first_error:
        # One retry with explicit correction guidance
        retry_hint = (
            f"Your previous response was not valid JSON matching the schema. "
            f"Error: {str(first_error)[:300]}. "
            f"Return ONLY the corrected raw JSON object, nothing else."
        )
        raw_retry = _call_llm(resume_text, retry_hint=retry_hint)
        cleaned_retry = _strip_markdown_fences(raw_retry)

        try:
            data = json.loads(cleaned_retry)
            parsed = ParsedResume(**data)
            parsed.raw_text = resume_text
            return parsed
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise RuntimeError(
                f"Resume parsing failed after retry. "
                f"First error: {first_error}. Second error: {second_error}."
            )
