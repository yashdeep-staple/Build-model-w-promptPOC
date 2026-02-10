from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime
import json

from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()
client = genai.Client()

sessions = {}  # session_id -> metadata


class CreateSessionResponse(BaseModel):
    session_id: str


class RecommendRequest(BaseModel):
    session_id: str
    prompt: str
    file_path: str


class RecommendResponse(BaseModel):
    session_id: str
    recommendations: dict

@app.post("/session", response_model=CreateSessionResponse)
def create_session():
    session_id = str(uuid4())
    sessions[session_id] = {
        "created_at": datetime.utcnow(),
        "file_id": None
    }
    return {"session_id": session_id}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_fields(req: RecommendRequest):

    # 1. Validate session
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    # 2. Upload file (scoped to this request)
    uploaded_file = client.files.upload(file=req.file_path)

    # Store file_id for reference (optional)
    sessions[req.session_id]["file_id"] = uploaded_file.name

    # 3. Agent prompt (recommendation mode)
    agent_prompt = """
    You are an AI assistant who will recommend entity and table names from the document that was uploaded

    Example:
    "Hey, from the document you uploaded here are some entities that can be extarcted:
    - InvoiceNumber
    -CustomerName
    -InvoiceDate

    If you want to extract any entities in particular please let me know."

    Please follow the above example.
    """

    # 4. Call Gemini
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[agent_prompt, uploaded_file],
    )

    recommendations = json.loads(response.text)

    return {
        "session_id": req.session_id,
        "recommendations": recommendations
    }

