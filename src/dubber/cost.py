"""
Cost estimation for different providers and services.
"""


def estimate_costs(
    audio_minutes: float,
    stt_mode: str,
    *,
    has_polish: bool,
    tts_provider: str,
    commentary: bool,
    commentary_ratio: float,
    rates: dict[str, float],
) -> dict[str, float]:
    """Estimate costs for the dubbing pipeline."""
    stt_cost = (
        audio_minutes * float(rates.get("stt_openai_per_min", 0.0)) if stt_mode == "openai" else 0.0
    )
    tts_minutes = audio_minutes * (1.0 + (float(commentary_ratio) if commentary else 0.0))
    tts_cost: float | None = None
    if tts_provider == "openai":
        rate = rates.get("tts_openai_per_min")
        if rate is not None:
            tts_cost = tts_minutes * float(rate)
    elif tts_provider == "elevenlabs":
        rate = rates.get("tts_elevenlabs_per_min")
        if rate is not None:
            tts_cost = tts_minutes * float(rate)
    polish_cost = 0.0
    if has_polish:
        tin = audio_minutes * float(rates.get("tokens_in_per_min", 200.0))
        tout = audio_minutes * float(rates.get("tokens_out_per_min", 200.0))
        polish_cost = (tin / 1_000_000.0) * float(rates.get("gpt_in_per_mtok", 0.60)) + (
            tout / 1_000_000.0
        ) * float(rates.get("gpt_out_per_mtok", 2.40))
    total = stt_cost + (tts_cost or 0.0) + polish_cost
    return {
        "stt_cost": stt_cost,
        "tts_cost": tts_cost,
        "polish_cost": polish_cost,
        "total": total,
        "tts_minutes": tts_minutes,
    }
