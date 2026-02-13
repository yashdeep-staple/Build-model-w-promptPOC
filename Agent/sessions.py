import uuid
import asyncio
from typing import Dict, Any

SESSION_STORE: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = asyncio.Lock()

async def get_or_create_session(session_id: str | None):
    async with SESSION_LOCK:
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = {"messages": []}

        return session_id, SESSION_STORE[session_id]

