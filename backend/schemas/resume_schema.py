"""
Structured schema for parsed resume data.
Every downstream agent (JD parser, fit scorer, gap analyzer) consumes this shape,
so keep it stable — if you add a field, add it with a default so old data doesn't break.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Education(BaseModel):
    degree: str
    institution: str
    field_of_study: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    cgpa_or_grade: Optional[str] = None


class Experience(BaseModel):
    title: str
    organization: str
    duration: Optional[str] = None
    description: Optional[str] = None
    skills_used: List[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    links: List[str] = Field(default_factory=list)  # github, linkedin, portfolio

    summary: Optional[str] = None

    skills: List[str] = Field(default_factory=list)
    tools_and_frameworks: List[str] = Field(default_factory=list)

    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)

    raw_text: Optional[str] = None  # kept for embedding/fit-scoring stage
