import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from agent import app_graph, extract_text_from_message
from sessions import get_or_create_session

app = FastAPI(title="StapleAI LangGraph Server")

# -------------------------
# REQUEST / RESPONSE
# -------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    usage: dict | None
    latency_ms: float | None

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id, session = await get_or_create_session(req.session_id)

    session["messages"].append(("user", req.message))
    inputs = {"messages": session["messages"]}

    final_message = None

    # LangGraph is sync → run in thread pool
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(
        None,
        lambda: list(app_graph.stream(inputs, stream_mode="values"))
    )

    for event in events:
        msg = event["messages"][-1]
        session["messages"].append(msg)
        if isinstance(msg, AIMessage):
            final_message = msg

    return ChatResponse(
        session_id=session_id,
        response=extract_text_from_message(final_message),
        usage=final_message.usage_metadata,
        latency_ms=final_message.response_metadata.get("latency_ms"),
    )

