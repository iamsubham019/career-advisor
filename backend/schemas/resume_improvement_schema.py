"""
Schema for the Resume Improvement Agent's output -- concrete rewritten
text the candidate can paste into their actual resume, not just abstract
advice (that's what gap_analysis.resume_wording_tips already covers).
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class BulletRewrite(BaseModel):
    section: str  # e.g. "Experience: IEEE EMBS Research Intern" or "Project: SkinScan AI"
    original: str
    improved: str
    reason: str  # why this rewrite helps for THIS specific JD


class ResumeImprovementResult(BaseModel):
    rewritten_summary: Optional[str] = None  # a JD-tailored professional summary, if resume lacks one or it's weak
    bullet_rewrites: List[BulletRewrite] = Field(default_factory=list)
    missing_skill_bullet_suggestions: List[str] = Field(default_factory=list)
    # net-new bullet ideas for skills the candidate has evidence of elsewhere but never wrote as a bullet
