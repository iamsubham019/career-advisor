"""
Final combined report returned by the single /analyze endpoint --
bundles every stage's output so the frontend can display as much or
as little detail as it wants without extra API calls.
"""

from pydantic import BaseModel

from schemas.resume_schema import ParsedResume
from schemas.jd_schema import ParsedJobDescription
from schemas.fit_schema import FitScoreResult
from schemas.gap_analysis_schema import GapAnalysisResult
from schemas.resume_improvement_schema import ResumeImprovementResult


class CareerAdvisorReport(BaseModel):
    parsed_resume: ParsedResume
    parsed_job_description: ParsedJobDescription
    fit_result: FitScoreResult
    gap_analysis: GapAnalysisResult
    resume_improvements: ResumeImprovementResult
