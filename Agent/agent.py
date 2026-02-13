import time
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from google import genai

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def analyze_document_for_entities(file_path: str) -> str:
    """
    Analyzes a document and recommends extractable entities and tables.
    """
    client = genai.Client()
    uploaded_file = client.files.upload(file=file_path)

    prompt = """
You are an AI assistant who will recommend entity and table names from the document that was uploaded.
"""

    start = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, uploaded_file],
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    print(f"[tool latency] analyze_document_for_entities: {latency_ms} ms")

    return response.text

tools = [analyze_document_for_entities]

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="""
You are an AI assistant for StapleAI that helps users build document extraction models.

Rules:
- Only call tools if a file path is provided.
- If no document is provided, ask for one.
- Never hallucinate document contents.
"""
    )

    start = time.perf_counter()
    response = model.invoke([system_prompt] + state["messages"])
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    response.response_metadata = response.response_metadata or {}
    response.response_metadata["latency_ms"] = latency_ms

    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "continue" if last.tool_calls else "end"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "agent")

app_graph = graph.compile()

def extract_text_from_message(message: BaseMessage) -> str:
    if isinstance(message.content, list):
        return " ".join(
            part.get("text", "")
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(message.content)

