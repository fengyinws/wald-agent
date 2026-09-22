from pydantic import ValidationError


def safe_cli_error(error: Exception) -> str:
    """Actionable field errors without Pydantic's echoed input (which may include keys)."""
    if isinstance(error, ValidationError):
        errors = error.errors(include_input=False, include_context=False, include_url=False)
        return "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'configuration'}: {item['msg']}"
            for item in errors[:4]
        )
    return str(error)
