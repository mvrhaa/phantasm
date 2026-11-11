"""Protocol version and the error codes sent in ``error`` messages."""

from enum import StrEnum

PROTOCOL_VERSION = 1


class ErrorCode(StrEnum):
    """Machine-readable reason carried by an ``error`` message."""

    UNAUTHORIZED = "unauthorized"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    UNKNOWN_TYPE = "unknown_type"
    INVALID_MESSAGE = "invalid_message"
    SESSION_NOT_FOUND = "session_not_found"
    TURN_BUSY = "turn_busy"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    TOOL_ERROR = "tool_error"
    ACTION_TIMEOUT = "action_timeout"
    CLIENT_GONE = "client_gone"
    INTERNAL = "internal"
