"""Hermes voice shim — bridges Home Assistant to Hermes Agent.

Receives text from HA's custom conversation agent, routes it through
the voice profile via one-shot mode (hermes -z), and returns the
response. No session persistence — each call is stateless, avoiding
context bloat and cold-start timing issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("hermes-shim")

app = FastAPI(title="Hermes Voice Shim")

HERMES = "/home/dshelfoon/.local/bin/hermes"
PROFILE = "voice"


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str


async def _hermes_oneshot(text: str) -> str:
    """Call hermes -z (one-shot) and return just the response text.

    One-shot mode bypasses TUI overhead — no banner, no spinner, no
    session parsing needed. Output is the agent's final text to stdout.
    """
    cmd = [HERMES, "-z", text, "-p", PROFILE]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    output = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        _LOGGER.error("Hermes failed (code %d): %s", proc.returncode, stderr_text)
        raise RuntimeError(f"Hermes exited with code {proc.returncode}")

    return output


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a voice command through Hermes one-shot mode."""
    _LOGGER.info("HA request: text=%r", request.text[:80])
    try:
        response_text = await _hermes_oneshot(request.text)
        return ChatResponse(response=response_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
