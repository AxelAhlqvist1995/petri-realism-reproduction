"""Helpers for OpenAI safety identifiers."""

import os

DEFAULT_OPENAI_SAFETY_IDENTIFIER = "Researcher_Axel_Ahlqvist_eval_experiments"
OPENAI_SAFETY_IDENTIFIER_ENV = "OPENAI_SAFETY_IDENTIFIER"
OPENAI_SAFETY_IDENTIFIER_MAX_LENGTH = 64


def get_openai_safety_identifier() -> str:
    """Return the stable OpenAI safety identifier for this evaluation workload."""
    identifier = (
        os.getenv(
            OPENAI_SAFETY_IDENTIFIER_ENV, DEFAULT_OPENAI_SAFETY_IDENTIFIER
        ).strip()
        or DEFAULT_OPENAI_SAFETY_IDENTIFIER
    )
    if len(identifier) > OPENAI_SAFETY_IDENTIFIER_MAX_LENGTH:
        raise ValueError(
            f"{OPENAI_SAFETY_IDENTIFIER_ENV} must be at most "
            f"{OPENAI_SAFETY_IDENTIFIER_MAX_LENGTH} characters"
        )
    return identifier
