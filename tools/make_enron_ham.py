# tools/make_enron_ham.py
#
# Lê data/raw/enron_spam_data.csv
# filtra apenas mensagens "ham" e grava em data/processed/enron_ham.csv
# no formato: id,label,subject,body,date

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "data" / "raw" / "enron_spam_data.csv"
DST_PATH = ROOT / "data" / "processed" / "enron_ham.csv"


def main():
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Ficheiro de origem não encontrado: {SRC_PATH}")

    df = pd.read_csv(SRC_PATH)

    expected_cols = {"Message ID", "Subject", "Message", "Spam/Ham", "Date"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas em falta em enron_spam_data.csv: {missing}")

    # Filtrar apenas ham
    df_ham = df[df["Spam/Ham"] == "ham"].copy()
    if df_ham.empty:
        raise ValueError("Nenhuma mensagem 'ham' encontrada no enron_spam_data.csv")

    out = pd.DataFrame(
        {
            "id": df_ham["Message ID"],
            "label": "ham",
            "subject": df_ham["Subject"].fillna(""),
            "body": df_ham["Message"].fillna(""),
            "date": df_ham["Date"],
        }
    )

    out = out.sort_values(["date", "id"]).reset_index(drop=True)

    DST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST_PATH, index=False)

    print(f"Enron HAM guardado em: {DST_PATH}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
