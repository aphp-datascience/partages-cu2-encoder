"""Turn a coded stays file into a training file for the two heads.

    uv run python -m scripts.prepare_heads --model models/cu2-icd10-dp-encoder \
        --src my_train.csv --out data/my_train

Input: the stays file of the PARTAGES guidelines (`scripts/stays.py`): `id`,
`text`, `procedure`, `primary_diag` (the PMSI DP, required here). A DP that is
not a Z code trains the `affection` head (`mdp` = Z769); a Z code as DP (e.g. a
chemotherapy session Z511) trains the `mdp` head, and the `affection` head is
masked for that stay (the file has no DR). The CCAM codes are written as the
first line of the text, as in training (`--no-acts` to skip).
Output: `<out>/data.parquet` with `note_id`, `note_text`, `affection`, `mdp`,
ready for `configs/finetune.yml`. A code outside the model's label space masks
its head and is counted, so the head sizes never change.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.heads import DEFAULT_MDP, resolve_model, split_targets
from scripts.stays import read_stays


def model_labels(model: str) -> dict[str, set[str]]:
    path = Path(resolve_model(model))
    vocab = json.loads((path / "doc_classifier" / "vocab.json").read_text())
    return {name: set(h["labels"]) for name, h in vocab["heads"].items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--src", required=True, type=Path, help="stays file (CSV or parquet)"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-acts", action="store_true", help="do not add the CCAM line")
    args = ap.parse_args()

    labels = model_labels(args.model)
    df = read_stays(args.src)
    if "primary_diag" not in df:
        raise SystemExit(
            f"{args.src}: no column primary_diag (the PMSI DP), needed for fine-tuning"
        )
    df = df[df["primary_diag"].notna()]
    heads = [split_targets(dp, None) for dp in df["primary_diag"]]
    mdp = [m if m in labels["mdp"] else None for m, _ in heads]
    aff = [a if a in labels["affection"] else None for _, a in heads]
    oov = Counter()
    for (_m, a), m2, a2 in zip(heads, mdp, aff, strict=True):
        oov["mdp"] += m2 is None
        oov["affection"] += a is not None and a2 is None

    texts = df["text"]
    if not args.no_acts:  # CCAM line on top of the text, as in training
        from scripts.acts import load_ccam, with_acts

        ccam = load_ccam()
        texts = [
            with_acts(t, a, ccam) for t, a in zip(texts, df["procedure"], strict=True)
        ]
    out = pd.DataFrame(
        {
            "note_id": df["id"].to_numpy(),
            "note_text": list(texts),
            "affection": aff,
            "mdp": mdp,
        }
    )
    out = out[out["affection"].notna() | out["mdp"].notna()]
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "data.parquet", index=False)
    print(
        f"{len(out)}/{len(df)} stays -> {args.out / 'data.parquet'} ; "
        f"DP in Z: {sum(m not in (None, DEFAULT_MDP) for m in mdp)} ; "
        f"outside the label space (masked): mdp {oov['mdp']}, "
        f"affection {oov['affection']}"
    )


if __name__ == "__main__":
    main()
