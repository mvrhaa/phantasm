"""``/browser``: where Zero's Chrome extension connects (``zero/browser.py``).

Only a Chrome extension may connect (its Origin is ``chrome-extension://…``),
and only with the token, given as ``?token=`` since an extension cannot set
headers on a WebSocket.
"""

import logging

from fastapi import APIRouter, WebSocket

from zero.server.auth import verify
from zero.server.state import get_state
from zero.server.ws import CLOSE_FORBIDDEN_ORIGIN, CLOSE_GOING_AWAY, CLOSE_UNAUTHORIZED

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/browser")
async def browser_endpoint(websocket: WebSocket) -> None:
    state = get_state(websocket.app)
    await websocket.accept()
    origin = websocket.headers.get("origin") or ""
    if not origin.startswith("chrome-extension://"):
        logger.warning("Rejected a browser connection that is not from an extension")
        await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN, reason="origin not allowed")
        return
    if not verify(websocket.query_params.get("token"), state.token):
        logger.warning("Rejected a browser connection with a missing or wrong token")
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="unauthorized")
        return
    if state.stopping:
        await websocket.close(code=CLOSE_GOING_AWAY, reason="shutting down")
        return
    await state.browser.attach(websocket)
