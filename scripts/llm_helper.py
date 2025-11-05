# scripts/llm_helper.py
from __future__ import annotations
import os
from typing import Tuple, Dict
from dotenv import load_dotenv

# Load .env once on import (expects HF_TOKEN and HF_MODEL in project root .env)
load_dotenv()

def _priority_levers(features: Dict, interp: Dict) -> list[str]:
    """Return improvement levers ranked by likely impact from metrics."""
    v = features.get("value", {})
    c = features.get("composition", {})
    col = features.get("color", {})

    # Value / contrast
    contrast = float(v.get("contrast", 0.15))
    dark_pct = float(v.get("dark_pct", 0.05))
    light_pct = float(v.get("light_pct", 0.02))

    # Composition / attention
    entropy_n = float(c.get("entropy_n", 0.5))
    focal     = float(c.get("focal_strength", 1.0))
    border    = float(c.get("border_pull", 0.0))

    # Color
    sat_mean  = float(col.get("saturation_mean", 0.2))

    levers = []

    # 1) Focus clarity first (biggest impact on read)
    if entropy_n > 0.65 or focal < 6.0:
        levers.append("focus_clarity")

    # 2) Global/local contrast (second biggest lever)
    low_light_anchors  = dark_pct < 0.05
    low_highlight_spark= light_pct < 0.03
    if contrast < 0.12 or low_light_anchors or low_highlight_spark:
        levers.append("contrast_structure")

    # 3) Edge pull (edge distractions)
    if border > 0.18:
        levers.append("edge_pull")

    # 4) Palette control / chroma management
    if sat_mean > 0.35 or sat_mean < 0.12:
        levers.append("palette_control")

    # Always dedupe & keep order
    seen = set(); out = []
    for k in levers:
        if k not in seen:
            seen.add(k); out.append(k)

    # If nothing triggered, still return something reasonable
    if not out:
        out = ["focus_clarity", "contrast_structure", "edge_pull", "palette_control"]

    return out


def build_coach_prompt(img, features: Dict, interp: Dict) -> Tuple[str, str]:
    v = features.get("value", {})
    c = features.get("composition", {})
    col = features.get("color", {})

    priorities = _priority_levers(features, interp)

    system_prompt = (
        "You are a helpful painting coach. Be specific, supportive, and actionable. "
        "Ground your critique in the provided metrics (value structure, attention behavior, color). "
        "Avoid rigid rules; focus on clarity, hierarchy, and viewer read."
    )

    user_prompt = f"""
Painting metrics (for your reasoning; don't repeat them verbatim):
- Value: mean={v.get('mean',0):.2f}, contrast={v.get('contrast',0):.2f}, dark%={(v.get('dark_pct',0)*100):.1f}, light%={(v.get('light_pct',0)*100):.1f}
- Composition: entropy_n={c.get('entropy_n',0):.2f}, focal_strength={c.get('focal_strength',0):.2f}, balance_lr={c.get('lr_balance',0):+.2f}, balance_tb={c.get('tb_balance',0):+.2f}, border_pull={c.get('border_pull',0):.2f}
- Color: saturation_mean={col.get('saturation_mean',0):.2f}, harmony={col.get('harmony','mixed')}, top_colors={col.get('palette', [])[:3]}

Rule-based summary (for your context; don't restate it): {interp.get('summary','')}

PRIORITY LEVERS (use the first two that apply most): {priorities}

Write a concise critique with **bullets on their own lines**:

FORMAT (strict):
LLM quick take
Strengths:
- <≤12 words, concrete>
- <≤12 words, concrete>
Improvements (top 2 levers only):
- Do: <one precise 10-minute action tied to metrics>
  Why: <one short reason about viewer clarity/hierarchy>
- Do: <one precise 10-minute action tied to metrics>
  Why: <one short reason about viewer clarity/hierarchy>

Rules:
- Do NOT restate the raw metrics or the summary sentence.
- Keep bullets short, one line each (wrap the Why on the next line).
- Choose the two actions that most improve read, based on PRIORITY LEVERS.
- No generic advice (e.g., “improve balance” alone is not allowed).
""".strip()

    return system_prompt, user_prompt



def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.5,
    top_p: float = 0.9,
) -> str:
    """
    Calls a Hugging Face hosted chat model using the Inference API.

    Requires:
      - HF_TOKEN  (in .env)
      - HF_MODEL  (in .env) e.g., "meta-llama/Llama-3.1-8B-Instruct" (or 70B)
    """
    REDACTED = os.getenv("HF_TOKEN")
    if not REDACTED:
        raise RuntimeError("Missing HF_TOKEN. Put it in your project .env (HF_TOKEN=...).")

    # Allow per-call override; otherwise use env
    model = (model or os.getenv("HF_MODEL") or "").strip()
    if not model or "/" not in model:
        raise RuntimeError("HF_MODEL missing/invalid. Example: meta-llama/Llama-3.1-8B-Instruct")

    try:
        from huggingface_hub import InferenceClient
        from huggingface_hub.errors import HfHubHTTPError
    except Exception as e:
        raise RuntimeError(
            f"huggingface_hub import failed: {e!r} "
            "(install into THIS venv and ensure no local file named 'huggingface_hub.py')."
        ) from e

    client = InferenceClient(model=model, token=REDACTED)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            raise RuntimeError(
                "HF auth/access error. Check HF_TOKEN and ensure you have access to the model page."
            ) from e
        if code == 404:
            raise RuntimeError(
                f"Model repo not found: '{model}'. Double-check HF_MODEL spelling."
            ) from e
        raise

    # Robust extraction across client versions
    try:
        choice = resp.choices[0]
        msg = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
        return content.strip()
    except Exception:
        return str(resp).strip()
