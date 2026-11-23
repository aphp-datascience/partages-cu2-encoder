"""Read the stays file of the PARTAGES guidelines (the Codabench input format).

One row per hospital stay (RUM), CSV (or parquet) with the columns

    id            unique id of the stay
    speciality    specialty of the stay (not used by the model)
    text          the reports of the stay (CRC, CRH, CR-OP), concatenated;
                  `texte` is accepted too
    procedure     the CCAM codes of the stay, separated by spaces
                  (e.g. "NDPA014 NDPA008"); empty if none
    primary_diag  the PMSI principal diagnosis (DP) of the RUM, as coded
                  (e.g. "M6534", "Z511"); for evaluation and fine-tuning only

Reports separated by the `[...]` marker of the guidelines are rejoined with the
separator the model was trained with (two blank lines).
"""

import re
from pathlib import Path

import pandas as pd

MARKER = re.compile(r"\s*\[\.\.\.\]\s*")
SEPARATOR = "\n\n\n"


def norm_code(code) -> str | None:
    """`M65.34 ` -> `M6534`; empty or missing -> None."""
    if not isinstance(code, str) or not code.strip():
        return None
    return code.replace(".", "").strip().upper()


def read_stays(path: Path) -> pd.DataFrame:
    """The stays: `id`, `text`, `procedure` (maybe empty), `primary_diag` if present."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_parquet(path)
    if "text" not in df and "texte" in df:
        df = df.rename(columns={"texte": "text"})
    missing = {"id", "text"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path}: missing columns {sorted(missing)} "
            "(expected id, speciality, text, procedure, primary_diag)"
        )
    df["id"] = df["id"].astype(str)
    df["text"] = [MARKER.sub(SEPARATOR, str(t)).strip() for t in df["text"]]
    df["procedure"] = (
        df["procedure"].fillna("").astype(str) if "procedure" in df else ""
    )
    if "primary_diag" in df:
        df["primary_diag"] = df["primary_diag"].map(norm_code)
    return df
