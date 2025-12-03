# tools/make_nazario_phishing_from_csv.py
#
# Lê data/raw/nazario.csv (dataset Nazario em CSV)
# e produz data/processed/nazario_phishing.csv no formato:
#   id,label,subject,body,date
#
# Assume que todas as linhas são phishing.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "data" / "raw" / "nazario.csv"
DST_PATH = ROOT / "data" / "processed" / "nazario_phishing.csv"


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool) -> str | None:
    """Escolhe a primeira coluna existente entre os candidatos."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Nenhuma das colunas {candidates} encontrada em nazario.csv")
    return None


def main():
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Ficheiro nazario.csv não encontrado em: {SRC_PATH}")

    df = pd.read_csv(SRC_PATH)

    # Tentar descobrir subject/body/date de forma robusta
    subject_col = pick_column(
        df,
        ["subject", "Subject", "SUBJECT"],
        required=False,
    )
    body_col = pick_column(
        df,
        ["body", "Body", "text", "Text", "Message", "message", "email", "Email"],
        required=True,
    )
    date_col = pick_column(
        df,
        ["date", "Date", "DATE"],
        required=False,
    )

    print("Coluna de subject detetada:", subject_col)
    print("Coluna de body detetada:", body_col)
    print("Coluna de date detetada:", date_col)

    if subject_col is None:
        subject_series = pd.Series([""] * len(df), index=df.index)
    else:
        subject_series = df[subject_col].fillna("")

    body_series = df[body_col].fillna("")

    if date_col is None:
        date_series = pd.Series([None] * len(df), index=df.index)
    else:
        date_series = df[date_col]

    out = pd.DataFrame(
        {
            "id": df.index,  
            "label": "phishing",
            "subject": subject_series,
            "body": body_series,
            "date": date_series,
        }
    )

    DST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST_PATH, index=False)

    print(f"Nazario phishing guardado em: {DST_PATH}")
    print(out["label"].value_counts())
    print("Exemplo de linhas:")
    print(out.head())


if __name__ == "__main__":
    main()
