"""Hermes voice shim — bridges Home Assistant to Hermes Agent.

Receives text from HA's custom conversation agent, routes it through
the voice profile (which has homeassistant tools), and returns the
response. Maintains conversation sessions per room with auto-reset
after MAX_TURNS to prevent context bloat.
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
MAX_TURNS = 20  # Auto-reset session after this many turns


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn: int
    max_turns: int
    reset: bool = False


def _load_sessions() -> dict:
    """Load session map from disk."""
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text())
    return {}


def _save_sessions(sessions: dict) -> None:
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
        response_text = output.strip().split("\n")[-1] if output.strip() else ""

    return response_text, session_id


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a voice command through Hermes."""
    _LOGGER.info("HA request: session=%s text=%r", request.session_id, request.text[:80])
    sessions = _load_sessions()
    sid = request.session_id or "default"

    # Load or initialize session state
    session = sessions.get(sid, {})
    if isinstance(session, str):
        # Legacy format — migrate
        session = {"resume": session, "turns": 0}

    resume = session.get("resume") if isinstance(session, dict) else None
    turns = session.get("turns", 0) if isinstance(session, dict) else 0
    reset = False

    # Auto-reset if over turn limit
    if turns >= MAX_TURNS:
        _LOGGER.info("Session %s: %d turns, auto-resetting", sid, turns)
        resume = None
        turns = 0
        reset = True

    try:
        response_text, new_session = await _hermes_chat(request.text, resume)

        # Store updated session
        if new_session:
            sessions[sid] = {"resume": new_session, "turns": turns + 1}
            _save_sessions(sessions)

        return ChatResponse(
            response=response_text,
            session_id=sid,
            turn=turns + 1,
            max_turns=MAX_TURNS,
            reset=reset,
        )

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
