"""
Schema for the Fit Scorer's output: a numeric match score plus the
matched/missing skill breakdown that the Gap Analysis Agent will explain.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SkillMatch(BaseModel):
    jd_skill: str
    matched_resume_item: Optional[str] = None  # None if no good match found
    similarity: float  # 0.0 to 1.0
    is_matched: bool  # True if similarity crossed the match threshold


class FitScoreResult(BaseModel):
    overall_score: float  # 0-100, weighted: must-haves count more than nice-to-haves
    must_have_matches: List[SkillMatch] = Field(default_factory=list)
    nice_to_have_matches: List[SkillMatch] = Field(default_factory=list)

    matched_skill_count: int
    total_required_skill_count: int

    missing_must_haves: List[str] = Field(default_factory=list)  # JD skills with no good resume match
    extra_resume_skills: List[str] = Field(default_factory=list)  # resume skills not asked for by JD
