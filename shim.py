"""Hermes voice shim — bridges Home Assistant to Hermes Agent.

Receives text from HA's custom conversation agent, routes it through
the voice profile (which has homeassistant tools), and returns the
response. Maintains conversation sessions per room.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("hermes-shim")

app = FastAPI(title="Hermes Voice Shim")

HERMES = "/home/dshelfoon/.local/bin/hermes"
PROFILE = "voice"
SESSIONS_FILE = Path("/home/dshelfoon/.hermes/shim-sessions.json")


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


def _load_sessions() -> dict[str, str]:
    """Load session map from disk."""
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text())
    return {}


def _save_sessions(sessions: dict[str, str]) -> None:
    """Save session map to disk."""
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))


async def _hermes_chat(text: str, resume: str | None = None) -> tuple[str, str]:
    """Call hermes chat and return (response_text, session_id)."""
    cmd = [HERMES, "chat", "-p", PROFILE, "-q", text]
    if resume:
        cmd.extend(["--resume", resume])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    output = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        _LOGGER.error("Hermes failed: %s", stderr_text)
        raise RuntimeError(f"Hermes exited with code {proc.returncode}")

    # Extract session ID from output footer
    session_id = ""
    match = re.search(r"hermes --resume (\S+)", output)
    if match:
        session_id = match.group(1)

    # Extract response message (everything between "╭─" and "╰─")
    lines = output.split("\n")
    response_lines = []
    in_response = False
    for line in lines:
        if "Query:" in line and not in_response:
            continue
        if "╭" in line:
            in_response = True
            continue
        if "╰" in line:
            break
        if in_response:
            response_lines.append(line)

    response_text = "\n".join(response_lines).strip()
    if not response_text:
        # Fallback: take last non-empty lines before ╰
        response_text = output.strip().split("\n")[-1] if output.strip() else ""

    return response_text, session_id


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a voice command through Hermes."""
    sessions = _load_sessions()
    resume = sessions.get(request.session_id) if request.session_id else None

    try:
        response_text, new_session = await _hermes_chat(request.text, resume)

        # Store session mapping
        sid = request.session_id or "default"
        if new_session:
            sessions[sid] = new_session
            _save_sessions(sessions)

        return ChatResponse(response=response_text, session_id=sid)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/reset/{session_id}")
async def reset_session(session_id: str):
    """Clear conversation history for a session."""
    sessions = _load_sessions()
    sessions.pop(session_id, None)
    _save_sessions(sessions)
    return {"status": "cleared"}
