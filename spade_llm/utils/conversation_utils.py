"""Utilities for managing conversation identifiers."""

import uuid
from spade.message import Message


def generate_unique_id(prefix: str = "session") -> str:
    """Generate a unique identifier with optional prefix using UUID4."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _extract_name(jid) -> str:
    """Extract the username from a JID, removing domain."""
    jid_str = str(jid)
    return jid_str.split("@")[0] if "@" in jid_str else jid_str


def generate_conversation_id(msg: Message) -> str:
    """
    Generate a unique conversation ID from a SPADE message.
    Uses thread if available, otherwise generates: sendername_receivername_uuid
    """
    if msg.thread:
        return msg.thread
    
    sender_name = _extract_name(msg.sender)
    receiver_name = _extract_name(msg.to)
    base_id = f"{sender_name}_{receiver_name}"
    unique_id = generate_unique_id(base_id)
    
    msg.thread = unique_id
    return unique_id