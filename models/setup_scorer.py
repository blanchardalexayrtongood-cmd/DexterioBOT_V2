from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SetupContext:
    symbol: str
    direction: str  # "long" ou "short"
    has_htf_bias_confluence: bool
    has_liquidity_sweep: bool
    has_bos_in_direction: bool
    uses_fvg: bool
    in_killzone: bool
    environment_choppy: bool
    es_bias_direction: Optional[str] = None
    es_structure_clean: Optional[bool] = None
    multi_signal_confirmed: Optional[bool] = None
    confirmed_after_retouch: Optional[bool] = None
    conflicting_signals: Optional[bool] = None
    session_match: Optional[bool] = None
    structure_score: Optional[int] = None


@dataclass
class SetupScore:
    score: int
    grade: str  # "A+", "A", "B", "C", "D"
    tags: List[str]


def score_setup_cjr(ctx: SetupContext) -> SetupScore:
    score = 0
    tags: List[str] = []

    weights = {
        "htf_bias": 3,
        "liquidity_sweep": 2,
        "bos": 2,
        "fvg": 1,
        "structure_clean": 1,
        "env_bonus": 1,
        "choppy_penalty": -2,
        "es_confluence": 2,
        "es_divergent": -3,
        "multi_signal": 1,
        "confirmed_retouch": 1,
        "conflict_penalty": -2,
        "session_match": 1,
    }

    if ctx.has_htf_bias_confluence:
        score += weights["htf_bias"]
        tags.append("htf_bias_ok")
    else:
        score -= weights["htf_bias"]
        tags.append("htf_bias_x")

    if ctx.has_liquidity_sweep:
        score += weights["liquidity_sweep"]
        tags.append("sweep")
    else:
        tags.append("no_sweep")

    if ctx.has_bos_in_direction:
        score += weights["bos"]
        tags.append("bos_ok")
    else:
        score -= 1
        tags.append("bos_x")

    if ctx.uses_fvg:
        score += weights["fvg"]
        tags.append("fvg")
    else:
        tags.append("no_fvg")

    if ctx.es_structure_clean:
        score += weights["structure_clean"]
        tags.append("structure_clean")
    elif ctx.es_structure_clean is False:
        score -= 1
        tags.append("structure_messy")

    if ctx.environment_choppy:
        score += weights["choppy_penalty"]
        tags.append("choppy")
    else:
        score += weights["env_bonus"]
        tags.append("env_clean")

    if ctx.symbol == "NQ=F" and ctx.es_bias_direction:
        if ctx.direction == ctx.es_bias_direction:
            score += weights["es_confluence"]
            tags.append("es_confluence")
        else:
            score += weights["es_divergent"]
            tags.append("es_divergent")

    if ctx.multi_signal_confirmed:
        score += weights["multi_signal"]
        tags.append("multi_signal")

    if ctx.confirmed_after_retouch:
        score += weights["confirmed_retouch"]
        tags.append("retouch_confirmed")

    if ctx.conflicting_signals:
        score += weights["conflict_penalty"]
        tags.append("conflicting_signals")

    if ctx.session_match:
        score += weights["session_match"]
        tags.append("session_match")

    if ctx.structure_score is not None:
        score += ctx.structure_score
        tags.append(f"struct_score_{ctx.structure_score}")

    if score >= 9:
        grade = "A+"
    elif score >= 7:
        grade = "A"
    elif score >= 5:
        grade = "B"
    elif score >= 3:
        grade = "C"
    else:
        grade = "D"

    return SetupScore(score=score, grade=grade, tags=tags)


def should_trade_es(score: SetupScore) -> bool:
    return score.grade in ("A+", "A")


def should_trade_nq(score: SetupScore, daily_r_running: float, daily_r_limit: float = -2.0) -> bool:
    if daily_r_running <= daily_r_limit:
        return False
    return score.grade in ("A+", "A") and score.score >= 8
