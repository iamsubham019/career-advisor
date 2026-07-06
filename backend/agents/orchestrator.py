"""
Career Advisor Orchestrator.

Wires the 4 agents into a single LangGraph state graph:

  resume_text + jd_text
        |
        v
  [parse_resume_node] --> parsed_resume
        |
        v
  [parse_jd_node] --> parsed_jd
        |
        v
  [score_fit_node] --> fit_result
        |
        v
  [analyze_gap_node] --> gap_analysis
        |
        v
  final CareerAdvisorReport

Each node reads/writes to a shared state dict (CareerAdvisorState) so any
node can be swapped, re-ordered, or extended (e.g. adding a parallel
resume-improvement-suggestions branch later) without rewriting the others.

Error handling: if any node raises, LangGraph propagates the exception up
to whoever invoked the graph (the FastAPI endpoint), which converts it into
a proper HTTP error. We don't swallow errors here -- a partial, silently
broken report is worse than a clear failure.
"""

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from agents.resume_parser import parse_resume
from agents.jd_parser import parse_job_description
from agents.fit_scorer import score_fit
from agents.gap_analyzer import analyze_gap
from agents.resume_improver import improve_resume

from schemas.resume_schema import ParsedResume
from schemas.jd_schema import ParsedJobDescription
from schemas.fit_schema import FitScoreResult
from schemas.gap_analysis_schema import GapAnalysisResult
from schemas.resume_improvement_schema import ResumeImprovementResult


class CareerAdvisorState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    parsed_resume: Optional[ParsedResume]
    parsed_jd: Optional[ParsedJobDescription]
    fit_result: Optional[FitScoreResult]
    gap_analysis: Optional[GapAnalysisResult]
    resume_improvements: Optional[ResumeImprovementResult]


def parse_resume_node(state: CareerAdvisorState) -> CareerAdvisorState:
    parsed = parse_resume(state["resume_text"])
    return {"parsed_resume": parsed}


def parse_jd_node(state: CareerAdvisorState) -> CareerAdvisorState:
    parsed = parse_job_description(state["jd_text"])
    return {"parsed_jd": parsed}


def score_fit_node(state: CareerAdvisorState) -> CareerAdvisorState:
    result = score_fit(state["parsed_resume"], state["parsed_jd"])
    return {"fit_result": result}


def analyze_gap_node(state: CareerAdvisorState) -> CareerAdvisorState:
    result = analyze_gap(state["parsed_resume"], state["parsed_jd"], state["fit_result"])
    return {"gap_analysis": result}


def improve_resume_node(state: CareerAdvisorState) -> CareerAdvisorState:
    result = improve_resume(state["parsed_resume"], state["parsed_jd"], state["fit_result"])
    return {"resume_improvements": result}


def build_graph():
    graph = StateGraph(CareerAdvisorState)

    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("score_fit", score_fit_node)
    graph.add_node("analyze_gap", analyze_gap_node)
    graph.add_node("improve_resume", improve_resume_node)

    # Resume and JD parsing don't depend on each other, but LangGraph's
    # StateGraph runs nodes sequentially by default unless using parallel
    # branches explicitly -- keeping it sequential here for simplicity and
    # easier debugging; can parallelize later if latency becomes an issue.
    graph.set_entry_point("parse_resume")
    graph.add_edge("parse_resume", "parse_jd")
    graph.add_edge("parse_jd", "score_fit")

    # analyze_gap and improve_resume both only depend on fit_result, not on
    # each other -- run them as parallel branches from score_fit so their
    # two Groq calls happen concurrently instead of back-to-back, cutting
    # total pipeline latency roughly in half for this portion.
    graph.add_edge("score_fit", "analyze_gap")
    graph.add_edge("score_fit", "improve_resume")
    graph.add_edge("analyze_gap", END)
    graph.add_edge("improve_resume", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Lazy-compile so importing this module is cheap."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_full_analysis(resume_text: str, jd_text: str) -> CareerAdvisorState:
    """
    Main entry point: raw resume text + raw JD text -> full final state
    containing parsed_resume, parsed_jd, fit_result, and gap_analysis.
    """
    graph = get_graph()
    initial_state: CareerAdvisorState = {
        "resume_text": resume_text,
        "jd_text": jd_text,
    }
    final_state = graph.invoke(initial_state)
    return final_state
