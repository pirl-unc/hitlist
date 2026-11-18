"""Normalize membership selections without changing their scientific meaning."""

from collections.abc import Iterable


def normalize_filter_values(values, name: str) -> list[str] | None:
    """Split commas and trim scalar/list tokens; keep None distinct from [].

    Blank tokens are ignored. Non-string tokens are invalid rather than
    coerced into apparent gene, peptide or provenance identifiers. Internal
    whitespace and case are preserved (e.g. ``Homo sapiens``).
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    elif not isinstance(values, Iterable) or isinstance(values, (bytes, dict)):
        raise TypeError(f"{name} filter requires a string or iterable of strings")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} filter values must be strings; got {type(value).__name__}")
        result.extend(token.strip() for token in value.split(",") if token.strip())
    return result
