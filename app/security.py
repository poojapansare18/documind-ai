from __future__ import annotations

import hmac
import json
import logging
import os
import re
import time
import uuid

from fastapi.responses import JSONResponse

logger = logging.getLogger("documind.http")
logger.setLevel(logging.INFO)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


async def security_middleware(request, call_next):
    """Apply optional API-key protection, security headers, and request logging."""
    started = time.perf_counter()
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if REQUEST_ID_PATTERN.fullmatch(supplied_id) else uuid.uuid4().hex
    request.state.request_id = request_id

    configured_key = os.getenv("DOCUMIND_API_KEY")
    path = request.url.path
    protected_path = path == "/documents" or path.startswith("/documents/")
    if configured_key and protected_path:
        supplied_key = request.headers.get("X-API-Key", "")
        if not supplied_key or not hmac.compare_digest(supplied_key.encode("utf-8"), configured_key.encode("utf-8")):
            response = JSONResponse(
                status_code=401,
                content={"detail": "A valid API key is required to access document data."},
            )
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(json.dumps({
        "event": "http_request",
        "request_id": request_id,
        "method": request.method,
        "path": path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    }, ensure_ascii=False))
    return response
