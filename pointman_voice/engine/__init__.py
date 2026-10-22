"""The call engine: listening, ending turns, speaking and barge-in, on Pipecat.

The package calls `start_call` when a call opens and `end_call` when it ends. The engine never touches
the board: it hands what happened to `emit`, as {"kind": "transcript" | "turn" | "ended" | "block", …},
and the package posts it.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

Emit = Callable[[dict], Awaitable[None] | None]


class NotYet(RuntimeError):
    """Calls need the node's two-way call audio and the engine; this version has neither."""


async def start_call(call: dict, transport: Any, settings: dict, emit: Emit) -> None:
    raise NotYet("Voice calls aren't open yet: the call engine comes in a later version.")


async def end_call(call_id: str) -> None:
    return None
