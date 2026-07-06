"""
Schema for the Gap Analysis Agent's output -- the human-readable advice
layer that sits on top of the Fit Scorer's numbers.
"""

from pydantic import BaseModel, Field
from typing import List


class GapAnalysisResult(BaseModel):
    overall_verdict: str  # 1-2 sentence plain-language summary of the fit
    strengths: List[str] = Field(default_factory=list)  # what already matches well, and why it's relevant
    key_gaps: List[str] = Field(default_factory=list)  # what's missing, explained in context of the JD
    actionable_suggestions: List[str] = Field(default_factory=list)  # concrete next steps to close gaps
    resume_wording_tips: List[str] = Field(default_factory=list)  # specific phrasing suggestions for the resume
