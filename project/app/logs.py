"""
Write agent interaction logs to JSON files.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

LOG_DIR = Path(os.getenv("LOGS_DIRECTORY", "logs"))
LOG_DIR.mkdir(exist_ok=True)


def log_entry(agent, messages, source: str = "user") -> dict:
    tools: list[str] = []
    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source,
    }


def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _last_message_timestamp(entry: dict) -> datetime:
    ts = entry["messages"][-1]["timestamp"]
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    raise TypeError(f"Unexpected timestamp type: {type(ts)}")


def log_interaction_to_file(agent, messages, source: str = "user") -> Path:
    entry = log_entry(agent, messages, source)
    ts_obj = _last_message_timestamp(entry)
    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath
