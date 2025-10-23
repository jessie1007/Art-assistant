# scripts/llm_helper.py
from __future__ import annotations

import os, json
from typing import Dict, List, Optional

# optional dependency: pip install openai>=1.0
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # fallback to heuristics if unavailable


SYSTEM = (
    "You are a painting coach. You will receive VALUE analysis only (no pixels). "
    "Return exactly THREE lines:\n"
    "1) One short, specific positive observation about what is working well.\n"
    "2) One concise actionable tip.\n"
    "3) One concise actionable tip.\n"
    "Stay concrete (values, big shapes, dominance/hierarchy). Avoid jargon."
)

def _prompt(metrics: Dict) -> List[Dict]:
    """Format a compact prompt from metrics."""
    content = {
        "blocks": {
            "K": metrics.get("blocks_K"),
            "areas_top3": metrics.get("areas_top3"),  # e.g. [["#3", 0.42], ["#1", 0.29], ["#4", 0.18]]
        },
        "plan": {
            "K": metrics.get("plan_K"),
            "centers_0_255": metrics.get("plan_centers"),
        },
        "coverage": {
            "mean": round(metrics.get("mean", 0.0), 3),
            "contrast": round(metrics.get("contrast", 0.0), 3),
            "dark_pct": round(metrics.get("dark_pct", 0.0), 3),
            "light_pct": round(metrics.get("light_pct", 0.0), 3),
        },
        "notes": metrics.get("notes", ""),
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Metrics:\n" + json.dumps(content, ensure_ascii=False)},
    ]


def tips_from_metrics(metrics: Dict, cfg_llm: Optional[Dict]) -> List[str]:
    """
    Returns exactly THREE strings: [praise, tip1, tip2].
    If the LLM is disabled/unavailable/errors → falls back to heuristics.
    """
    # LLM disabled or client missing → heuristics
    if not cfg_llm or not cfg_llm.get("enabled", False) or OpenAI is None:
        return _heuristic_praise_and_tips(metrics)

    try:
        provider = (cfg_llm.get("provider") or "openai").lower()
        model = cfg_llm.get("model", "gpt-4o-mini")
        temperature = float(cfg_llm.get("temperature", 0.2))

        if provider == "openai":
            client = OpenAI()  # uses OPENAI_API_KEY
        else:
            # allows OpenAI-compatible local servers (set OPENAI_BASE_URL)
            client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", None),
                api_key=os.getenv("OPENAI_API_KEY", "nokey"),
            )

        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=_prompt(metrics),
            max_tokens=160,
        )
        txt = (resp.choices[0].message.content or "").strip()

        # Normalize output → exactly 3 lines
        lines = [l.strip(" \t-•") for l in txt.split("\n") if l.strip()]
        if len(lines) >= 3:
            return [lines[0], lines[1], lines[2]]

        # If the model returned a paragraph, split on sentences
        sentences = [s.strip() for s in txt.replace("\n", " ").split(". ") if s.strip()]
        if len(sentences) >= 3:
            return [sentences[0] + ".", sentences[1] + ".", sentences[2] + "."]

        # Fallback if formatting is odd
        return _heuristic_praise_and_tips(metrics)

    except Exception:
        return _heuristic_praise_and_tips(metrics)


# ---------------- Heuristics (fallback & augmentation) ----------------

def _heuristic_praise_and_tips(m: Dict) -> List[str]:
    """Bundle praise + two tips."""
    praise = _heuristic_praise(m)
    t1, t2 = _heuristic_tips(m)
    return [praise, t1, t2]

def _heuristic_praise(m: Dict) -> str:
    """
    Choose one positive, specific observation from simple metrics.
    Prefers clear dominance, then balanced families, then solid range.
    """
    mean_ = m.get("mean", 0.0)
    contrast = m.get("contrast", 0.0)
    light = m.get("light_pct", 0.0)
    areas = m.get("areas_sorted", [])  # [(gray, frac), ...] largest→smallest

    if areas:
        dom = areas[0][1]
        if dom >= 0.35:
            return f"Good dominance: your largest mass reads clearly (≈{dom*100:.0f}%)."
    if 0.45 <= light <= 0.65:
        return "Nice balance between light and dark families—neither is overpowering."
    if contrast >= 0.18:
        return "Solid value range—the lights and darks separate cleanly."
    return "Cohesive simplification—values sit in clear families overall."

def _heuristic_tips(m: Dict) -> List[str]:
    """
    Very lightweight coaching rules; returns two concise tips.
    """
    tips: List[str] = []

    # Dominance check: biggest vs second biggest too close in area & value
    areas_sorted = m.get("areas_sorted", [])  # [(gray, frac), ...] largest→smallest
    if len(areas_sorted) >= 2:
        gap = areas_sorted[0][1] - areas_sorted[1][1]
        # if plan centers available, approximate value gap using nearest center delta
        val_gap = None
        centers = m.get("plan_centers") or []
        if centers and len(centers) >= 2:
            # treat first two areas' gray as is (0..255); get absolute diff to nearest centers
            v0, v1 = areas_sorted[0][0], areas_sorted[1][0]
            val_gap = abs(int(v0) - int(v1)) / 255.0
        if val_gap is None:
            val_gap = 0.0  # be conservative

        if gap < 0.06 and val_gap < 0.08:
            tips.append("Make one mass clearly dominant—merge or adjust values so the lead shape reads at least ~30% of the canvas.")

    # Lights crowded?
    light_pct = m.get("light_pct", 0.0)
    if light_pct > 0.55:
        tips.append("Group your lights: fold small light breaks into the nearest larger light mass before adding accents.")

    # Contrast too low?
    if m.get("contrast", 0.0) < 0.12 and len(tips) < 2:
        tips.append("Push separation between the lightest light and the rest of the light family; keep darks unified.")

    # Ensure two tips
    default_tip = "Start with K=5 big value blocks; paint the largest mass first, then place a single opposite-value accent for hierarchy."
    while len(tips) < 2:
        tips.append(default_tip)

    return tips[:2]
