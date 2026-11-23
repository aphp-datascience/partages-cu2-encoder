"""PMSI (DP, DR) <-> the two heads of the CU2 encoder.

- `mdp` head ("mode de prise en charge"): the Z codes that can be a DP; `Z769` means
  "no Z code" (the DP is the disease itself).
- `affection` head: the disease or permanent state, i.e. the DR when the DP is a Z code,
  otherwise the DP. It covers the non-Z codes plus Z80-Z99 (history, transplant, implant
  and absent-organ states), which a DR may legitimately be.

A DP in Z without a DR has no affection: that head is masked (target None, ignored
by the loss). At inference, when `mdp` predicts a Z code, a small rule
(`data/dr_rule.json`, per Z code then per Z category) decides whether the affection
is kept as DR.
"""

import json
from pathlib import Path

DEFAULT_MDP = "Z769"


def resolve_model(model: str) -> str:
    """Local dir as is; a Hub repo id (`org/name` or `org/name@revision`, e.g.
    `PARTAGES-CU2/cu2-icd10-dp-encoder@<revision>`) is downloaded or read from the HF
    cache."""
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download

    repo, _, revision = model.partition("@")
    return snapshot_download(repo, revision=revision or None)


def is_z(code) -> bool:
    return isinstance(code, str) and code.startswith("Z")


def is_state(code) -> bool:
    """Z80-Z99: a permanent state that a DR may be (guide MCO, DR section)."""
    return is_z(code) and code[1:3].isdigit() and 80 <= int(code[1:3]) <= 99


def split_targets(dp: str, dr: str | None) -> tuple[str, str | None]:
    """PMSI (DP, DR) -> (mdp target, affection target); affection None = masked."""
    if not isinstance(dp, str) or not dp:
        raise ValueError("empty DP")
    dr = dr if isinstance(dr, str) and dr else None
    if not is_z(dp):
        return DEFAULT_MDP, dr or dp
    if dr is None or (is_z(dr) and not is_state(dr)):
        return dp, None
    return dp, dr


def load_rule(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def keeps_dr(mdp: str, rule: dict | None) -> bool:
    """Is the affection kept as DR when the `mdp` head predicts this Z code?"""
    if rule is None:
        return True
    if mdp in rule["code"]:
        return bool(rule["code"][mdp])
    return bool(rule["categorie"].get(mdp[:3], True))


def reconstruct(mdp, affection, rule: dict | None) -> tuple[str, str | None]:
    """Head outputs -> PMSI (DP, DR)."""
    if not isinstance(mdp, str) or mdp == DEFAULT_MDP:
        return affection, None
    if keeps_dr(mdp, rule):
        return mdp, affection
    return mdp, None


def dp_candidates(
    dp: str, aff_probs: dict, mdp_probs: dict, k: int
) -> list[tuple[str, float]]:
    """Top-k candidate DPs (the PMSI DP, Z codes included), ranked by their probability
    of being the DP: a Z code z by P_mdp(z); a disease a by P_mdp(no Z) x P_aff(a).
    The predicted DP comes first."""
    p_none = mdp_probs.get(DEFAULT_MDP, 0.0)
    score = {a: p * p_none for a, p in aff_probs.items()}
    for z, p in mdp_probs.items():
        if z != DEFAULT_MDP:
            score[z] = max(score.get(z, 0.0), p)
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    out = [(dp, score.get(dp, 0.0))]
    out += [(c, s) for c, s in ranked if c != dp]
    return out[:k]
