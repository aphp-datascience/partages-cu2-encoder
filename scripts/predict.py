"""Predict the principal diagnosis (DP) of hospital stays with the CU2 encoder.

    uv run python -m scripts.predict --model models/cu2-icd10-dp-encoder \
        --data stays.csv --out preds.csv [--topk 10]

Input: the stays file of the PARTAGES guidelines (`scripts/stays.py`): CSV with
`id`, `speciality`, `text`, `procedure`, `primary_diag` (the last one optional).
The CCAM codes of `procedure` are written as the first line of the text, as in
training (`--no-acts` to skip).
Output (CSV, or parquet if --out ends with .parquet): one row per stay with the
predicted PMSI coding (`primary_diag_pred` = the DP, `dr`), the raw head outputs
(`affection`, `mdp`) and, with
`--topk K`, the K most likely DP codes (`candidates`, `scores`).
If `primary_diag` is given, also prints the accuracy of the predicted DP against
it: exact code, first 4 characters (the Codabench metric), first 3 characters,
and top-K.
"""

import argparse
from pathlib import Path

import edsnlp

from scripts.heads import (
    DEFAULT_MDP,
    dp_candidates,
    load_rule,
    reconstruct,
    resolve_model,
)
from scripts.stays import read_stays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local dir")
    ap.add_argument(
        "--data", required=True, type=Path, help="stays file (CSV or parquet)"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--rule",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "dr_rule.json",
        help="keep-the-DR rule per Z code; 'none' always keeps the affection as DR",
    )
    ap.add_argument("--no-acts", action="store_true", help="do not add the CCAM line")
    ap.add_argument("--topk", type=int, default=0, help="also output K candidates")
    ap.add_argument("--num-cpu", type=int, default=4, help="CPU workers (tokenisation)")
    args = ap.parse_args()
    rule = None if str(args.rule) == "none" else load_rule(args.rule)

    stays = read_stays(args.data)
    texts = stays["text"]
    if not args.no_acts:  # CCAM line on top of the text, as in training
        from scripts.acts import load_ccam, with_acts

        ccam = load_ccam()
        texts = [
            with_acts(t, a, ccam)
            for t, a in zip(texts, stays["procedure"], strict=True)
        ]
    frame = (
        stays[["id"]].rename(columns={"id": "note_id"}).assign(note_text=list(texts))
    )

    nlp = edsnlp.load(resolve_model(args.model))
    if args.topk:  # keep the probabilities of each head's top labels
        pipe = nlp.get_pipe("doc_classifier")
        pipe.store_probs, pipe.probs_topk = True, max(args.topk, 20)
    docs = (
        edsnlp.data.from_pandas(frame, tokenizer=nlp.tokenizer, converter="omop")
        .map_pipeline(nlp)
        .set_processing(num_cpu_workers=args.num_cpu, show_progress=True)
    )
    df = (
        docs.to_pandas(
            converter="omop",
            doc_attributes=["affection", "mdp", *(["prob"] * bool(args.topk))],
        )
        .set_index("note_id")
        .loc[frame["note_id"]]
        .reset_index()
    )

    pairs = [
        reconstruct(m, a, rule) for m, a in zip(df["mdp"], df["affection"], strict=True)
    ]
    out = stays[["id"]].copy()
    out["primary_diag_pred"] = [p[0] for p in pairs]
    out["dr"] = [p[1] for p in pairs]
    out["affection"], out["mdp"] = df["affection"].to_numpy(), df["mdp"].to_numpy()
    if (
        args.topk
    ):  # DP candidates: Z codes by the mdp head, diseases by P(no Z) x P(disease)
        lists = [
            dp_candidates(
                dp,
                (p or {}).get("affection") or {},
                (p or {}).get("mdp") or {},
                args.topk,
            )
            for dp, p in zip(out["primary_diag_pred"], df["prob"], strict=True)
        ]
        out["candidates"] = ["|".join(c for c, _ in cs) for cs in lists]
        out["scores"] = ["|".join(f"{s:.4g}" for _, s in cs) for cs in lists]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".parquet":
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    n_z = int((out["mdp"] != DEFAULT_MDP).sum())
    print(f"{len(out)} stays -> {args.out} ({n_z} with a Z code as DP)")

    if "primary_diag" in stays:
        g = stays["primary_diag"]
        s = g.notna()
        pred = out.loc[s, "primary_diag_pred"].fillna("")
        msg = (
            f"predicted DP vs primary_diag, n={int(s.sum())}: "
            f"exact {(pred == g[s]).mean():.2%}, "
            f"4 characters {(pred.str[:4] == g[s].str[:4]).mean():.2%}, "
            f"3 characters {(pred.str[:3] == g[s].str[:3]).mean():.2%}"
        )
        if args.topk:
            top = [
                gg in c.split("|")
                for gg, c in zip(g[s], out.loc[s, "candidates"], strict=True)
            ]
            msg += f", top-{args.topk} {sum(top) / len(top):.2%}"
        print(msg)


if __name__ == "__main__":
    main()
