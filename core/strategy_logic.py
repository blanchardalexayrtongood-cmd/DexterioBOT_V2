# core/strategy_logic.py

from typing import Tuple, Optional


def decide_direction_and_context(
    symbol: str,
    logic_mode: str,
    bos_long: bool,
    bos_short: bool,
    fvg_long: bool,
    fvg_short: bool,
    sweep_long: bool,
    sweep_short: bool,
    smt_long_ok: bool = False,
    smt_short_ok: bool = False,
) -> Tuple[Optional[str], str, str]:
    """
    Détermine la direction potentielle à trader selon les signaux.
    Retourne:
      - direction: "long", "short" ou None
      - context: tag de confluence (ex: "bos,sweep,fvg")
      - smt_tag: "smt_long_ok" / "smt_short_ok" / ""
    """
    context_tags = []
    smt_tag = ""
    direction: Optional[str] = None

    if logic_mode == "smt_only":
        if smt_long_ok:
            direction = "long"
            smt_tag = "smt_long_ok"
        elif smt_short_ok:
            direction = "short"
            smt_tag = "smt_short_ok"
        return direction, smt_tag, smt_tag

    # Mode pro : on veut bos + fvg/sweep, SMT en bonus si dispo
    if logic_mode in ("full_trader_pro",):
        if bos_long and (fvg_long or sweep_long):
            if smt_long_ok or not (smt_long_ok or smt_short_ok):
                direction = "long"
                context_tags = ["bos"]
                if fvg_long: context_tags.append("fvg")
                if sweep_long: context_tags.append("sweep")
                if smt_long_ok: smt_tag = "smt_long_ok"

        elif bos_short and (fvg_short or sweep_short):
            if smt_short_ok or not (smt_long_ok or smt_short_ok):
                direction = "short"
                context_tags = ["bos"]
                if fvg_short: context_tags.append("fvg")
                if sweep_short: context_tags.append("sweep")
                if smt_short_ok: smt_tag = "smt_short_ok"

    else:
        # Mode permissif : un seul élément déclenche
        if bos_long or fvg_long or sweep_long:
            direction = "long"
            if bos_long: context_tags.append("bos")
            if fvg_long: context_tags.append("fvg")
            if sweep_long: context_tags.append("sweep")
        elif bos_short or fvg_short or sweep_short:
            direction = "short"
            if bos_short: context_tags.append("bos")
            if fvg_short: context_tags.append("fvg")
            if sweep_short: context_tags.append("sweep")

    context = ",".join(context_tags)
    return direction, context, smt_tag
