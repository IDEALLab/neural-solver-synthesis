"""Fail-closed prompt perturbations for SDS evaluation experiments."""

from __future__ import annotations

import hashlib

CANONICAL_VARIANT = "canonical"
WITHOUT_HYPOTHESIZE_VARIANT = "without-hypothesize"
PROMPT_VARIANTS = (CANONICAL_VARIANT, WITHOUT_HYPOTHESIZE_VARIANT)

HYPOTHESIZE_INSTRUCTION = (
    "- **Hypothesize**: Propose a search strategy capable of exploring the solution "
    "space effectively. Consider how to balance \"exploitation\" (improving a good "
    "solution) with \"exploration\" (escaping bad local optima).\n"
)


class PromptVariantError(ValueError):
    """Raised when a frozen prompt perturbation cannot be applied exactly."""

    def __init__(
        self,
        *,
        variant: str | None = None,
        occurrence_count: int | None = None,
    ) -> None:
        if variant is not None:
            message = f"Unsupported prompt variant: {variant}"
        else:
            message = (
                "Expected exactly one frozen Hypothesize instruction, found "
                f"{occurrence_count}"
            )
        super().__init__(message)


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def apply_prompt_variant(system_prompt: str, variant: str) -> str:
    """Apply one predeclared prompt perturbation without changing other bytes."""
    if variant == CANONICAL_VARIANT:
        return system_prompt
    if variant != WITHOUT_HYPOTHESIZE_VARIANT:
        raise PromptVariantError(variant=variant)

    occurrence_count = system_prompt.count(HYPOTHESIZE_INSTRUCTION)
    if occurrence_count != 1:
        raise PromptVariantError(occurrence_count=occurrence_count)
    return system_prompt.replace(HYPOTHESIZE_INSTRUCTION, "", 1)
