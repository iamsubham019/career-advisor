"""
Structured schema for parsed job description data.
Mirrors the resume schema's granularity so the Fit Scorer can compare
"required skills" against "candidate skills" apples-to-apples.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ParsedJobDescription(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None  # e.g. "Full-time", "Internship", "Contract"

    experience_level: Optional[str] = None  # e.g. "Entry-level", "2-4 years", "Senior"

    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    tools_and_frameworks: List[str] = Field(default_factory=list)

    responsibilities: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)  # degree, certifications required

    summary: Optional[str] = None

    raw_text: Optional[str] = None  # kept for embedding/fit-scoring stage
