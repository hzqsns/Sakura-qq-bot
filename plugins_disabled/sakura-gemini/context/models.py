import time
from dataclasses import dataclass


@dataclass
class ContextMessage:
    """A single message recorded in the conversation context."""
    sender_id: str
    sender_name: str
    content: str
    image_urls: list[str]
    timestamp: float
    is_bot_reply: bool

    def is_expired(self, expire_seconds: int) -> bool:
        """Return True if this message is older than expire_seconds."""
        return (time.time() - self.timestamp) > expire_seconds
