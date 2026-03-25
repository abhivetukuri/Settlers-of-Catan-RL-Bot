from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedActions:
    signatures: list[str]
    action_to_index: dict[str, int]


def action_signature(action) -> str:
    return f"{action.color.value}|{action.action_type.name}|{repr(action.value)}"


def encode_legal_actions(playable_actions) -> EncodedActions:
    actions = list(playable_actions)
    signatures = sorted(action_signature(action) for action in actions)
    action_to_index = {signature: i for i, signature in enumerate(signatures)}
    return EncodedActions(signatures=signatures, action_to_index=action_to_index)
