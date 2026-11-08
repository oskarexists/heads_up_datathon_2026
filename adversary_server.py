"""HTTP server for one review adversary.

    ADVERSARY_STYLE=maniac|calling_station|min_raise|exploiter
    PORT defaults to 5009.

The routes match baseline-bots/ so judge_engine.py can drive the process.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from game import Action, AgentConfig, HandResult, Observation
from review_adversaries import ADVERSARIES

STYLE = os.environ.get("ADVERSARY_STYLE", "calling_station")
if STYLE not in ADVERSARIES:
    raise SystemExit(f"ADVERSARY_STYLE must be one of {sorted(ADVERSARIES)}, got {STYLE!r}")

BOT = ADVERSARIES[STYLE]()
PARTICIPANT = "Adversary Review"
AGENT_NAME = STYLE


def new_half(config: AgentConfig, memory: dict) -> None:
    BOT.new_half(config, memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    return BOT.choose_action(obs, memory)


def on_hand_end(result: HandResult, memory: dict) -> None:
    BOT.on_hand_end(result, memory)


app = FastAPI()
MEMORY: dict = {}
STATUS = {"started": False}
TOKEN = os.environ.get("AGENT_TOKEN") or None


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _deliver_results(raw_results) -> list:
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return JSONResponse(_error_info(exc), status_code=500)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors}, status_code=500)
    answer = action.name if isinstance(action, Action) else action
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
