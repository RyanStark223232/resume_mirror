import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import os, getpass

from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

# ---- Structured output models ----
class Qualifications(BaseModel):
    required: List[str] = Field(
        description="List of required qualifications for the job."
    )
    preferred: List[str] = Field(
        description="List of preferred qualifications for the job."
    )

class JobQualsState(TypedDict):
    job_post: str   # Raw job posting text
    human_feedback: str   # Human feedback on missing qualifications
    qualifications: Qualifications # Extracted qualifications

# ---- Instructions ----
qualification_instructions = """You are tasked with analyzing a job posting and coming up with a outline for writing tailored resume.

1. Carefully read the following job post:
{job_post}

2. Extract qualifications into two categories, focus on keywords of technology required, such as Power BI and AWS :
   - Required (must-have skills/credentials/experience)
   - Preferred (nice-to-have or optional)

3. Update qualification with similar item based on human feedback of the user:
{human_feedback}

3. Return the structured lists.
"""

# ---- Node: Extract qualifications ----
def extract_qualifications(state: JobQualsState):
    job_post = state["job_post"]
    human_feedback = state.get("human_feedback", "")

    structured_llm = llm.with_structured_output(Qualifications)

    system_message = qualification_instructions.format(
        job_post=job_post, 
        human_feedback = human_feedback
    )

    qualifications = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Extract the qualifications now.")]
    )

    return {"qualifications": qualifications}

# ---- Node: Human feedback ----
def human_feedback_node(state: JobQualsState):
    """Interrupts the flow to allow a human to provide feedback."""
    pass

# ---- Graph construction ----
qua_builder = StateGraph(JobQualsState)
qua_builder.add_node("extract_qualifications", extract_qualifications)

qua_builder.add_edge(START, "extract_qualifications")
qua_builder.add_edge("extract_qualifications", END)

# --- Outer graph state ---
class ResumeState(TypedDict):
    job_post: str
    human_feedback: str
    resume_input: str
    qualifications: Qualifications
    resume_draft: str
    editor_feedback: Annotated[List[str], operator.add]  # accumulate feedback
    iteration: int

# --- Step 2: Draft resume ---
def draft_resume(state: ResumeState):
    job_post = state["job_post"]
    qualifications = state["qualifications"]
    resume_input = state["resume_input"]
    feedback = "\n".join(state.get("editor_feedback", []))

    system_message = f"""You are a career coach.
    Tailor the candidate's resume to this job post:

    Job Post:
    {job_post}

    Required Qualifications:
    {qualifications.required}

    Preferred Qualifications:
    {qualifications.preferred}

    Existing Resume:
    {resume_input}

    Previous Draft (if any):
    {state.get("resume_draft", "")}

    Feedback from reviewers:
    {feedback}

    Draft a revised resume highlighting relevant achievements, quantifying impact where possible.
    """
    draft = llm.invoke([SystemMessage(content=system_message)])
    return {"resume_draft": draft.content, "editor_feedback": []}  # clear feedback after applying

# --- Step 3a: Editor 1 (API stub) ---
def editor_api(state: ResumeState):
    return {"editor_feedback": ["[API editor returned no feedback]"]}

# --- Step 3b: Editor 2 (LLM critique) ---
def editor_critique(state: ResumeState):
    draft = state["resume_draft"]
    system_message = """You are a strict resume reviewer.
    Critique the resume draft for:
    - Grammar issues
    - Lack of quantified metrics
    - Weak descriptions of achievements
    Provide actionable suggestions.
    """
    feedback = llm.invoke([SystemMessage(content=system_message), HumanMessage(content=draft)])
    return {"editor_feedback": [feedback.content]}

# ---- Control: continue or end ----
def should_human_feedback_continue(state: JobQualsState):
    human_feedback = state.get("human_feedback")
    if (human_feedback is None) or (human_feedback == "ok"):
        return "draft_resume"
    return "extract_qualifications"

# --- Step 3c: Router node to parallel editors ---
def send_to_editors(state: ResumeState):
    draft = state["resume_draft"]
    return [
        Send("editor_api", {"resume_draft": draft}),
        Send("editor_critique", {"resume_draft": draft})
    ]

# --- Step 4: Revise resume ---
def revise_resume(state: ResumeState):
    # increment iteration and let feedback flow back to draft_resume
    return {
        "iteration": state.get("iteration", 0) + 1
    }

# --- Step 5: Route based on iteration count ---
def should_continue(state: ResumeState):
    if state.get("iteration", 0) >= 2:
        return "write_final_draft"
    return "draft_resume"

# --- Step 6: Final Draft Node ---
def write_final_draft(state: ResumeState):
    """
    Produce the polished final resume using the last revised draft
    and all accumulated feedback.
    """
    draft = state.get("resume_draft", "")
    feedback = "\n".join(state.get("editor_feedback", []))

    system_message = f"""You are a professional career coach.
    Produce a final, polished resume based on the following:

    Last Draft:
    {draft}

    Feedback from editors (if any):
    {feedback}

    Ensure:
    - Clear grammar and style
    - Achievements quantified
    - Strong focus on required and preferred qualifications
    """
    final_resume = llm.invoke([SystemMessage(content=system_message)])
    return {"resume_draft": final_resume.content}

# --- Build main graph ---
builder = StateGraph(ResumeState)

builder.add_node("extract_qualifications", qua_builder.compile())
builder.add_node("human_feedback", human_feedback_node)
builder.add_node("draft_resume", draft_resume)
builder.add_node("editor_api", editor_api)
builder.add_node("editor_critique", editor_critique)
builder.add_node("revise_resume", revise_resume)
builder.add_node("write_final_draft", write_final_draft, description="Produce polished final resume")

builder.add_edge(START, "extract_qualifications")
builder.add_edge("extract_qualifications", "human_feedback")
builder.add_conditional_edges("human_feedback", should_human_feedback_continue, ["extract_qualifications", "draft_resume"])
builder.add_edge("draft_resume", "editor_api")
builder.add_edge("draft_resume", "editor_critique")
builder.add_edge("editor_api", "revise_resume")
builder.add_edge("editor_critique", "revise_resume")
builder.add_conditional_edges("revise_resume", should_continue, ["draft_resume", "write_final_draft"])
builder.add_edge("write_final_draft", END)

# Compile
graph = builder.compile(interrupt_before=["human_feedback"])