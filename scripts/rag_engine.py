import json, math
from pathlib import Path

def _passes(rule, facts):
    # rule like {"thirds_score<":0.45}
    for k, t in rule.items():
        if k.endswith("<"):
            key = k[:-1];    ok = float(facts.get(key, math.inf)) < float(t)
        elif k.endswith(">"):
            key = k[:-1];    ok = float(facts.get(key, -math.inf)) > float(t)
        elif k.endswith("<="):
            key = k[:-2];    ok = float(facts.get(key, math.inf)) <= float(t)
        elif k.endswith(">="):
            key = k[:-2];    ok = float(facts.get(key, -math.inf)) >= float(t)
        else:  # equality or boolean flags
            ok = str(facts.get(k)) == str(t)
        if not ok: return False
    return True

def load_tips(path="data/tips/tips.jsonl"):
    tips = []
    p = Path(path)
    if not p.exists(): return tips
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            tips.append(json.loads(line))
    return tips

def retrieve_tips(facts: dict, tips_db):
    # facts: {"thirds_score":..., "entropy":..., "sat_mean":..., "focal_local_contrast":...}
    matched = []
    for tip in tips_db:
        when = tip.get("when", {})
        if _passes(when, facts):
            matched.append(tip["tip"])
    # Dedup & capß
    out, seen = [], set()
    for t in matched:
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:6]


