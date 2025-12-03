# tools/prepare_enron_csv.py

import pandas as pd
from pathlib import Path

# Caminho para o CSV original 
SRC_PATH = Path("data/raw/enron_spam_data.csv")
DST_DIR = Path("data/processed")
DST_DIR.mkdir(parents=True, exist_ok=True)
DST_PATH = DST_DIR / "emails_full.csv"

def main():
    df = pd.read_csv(SRC_PATH)

    # Verificações básicas
    expected_cols = {"Message ID", "Subject", "Message", "Spam/Ham", "Date"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV de origem: {missing}")

    # Mapeamento de labels
    label_map = {
        "ham": "ham",
        "spam": "phishing",
    }
    df["label"] = df["Spam/Ham"].map(label_map)

    # verifica se existe algum valor inesperado
    if df["label"].isna().any():
        bad_vals = df.loc[df["label"].isna(), "Spam/Ham"].unique()
        raise ValueError(f"Valores de 'Spam/Ham' inesperados: {bad_vals}")

    # Construi o dataframe final no formato do projeto
    out = pd.DataFrame(
        {
            "id": df["Message ID"],
            "label": df["label"],
            "subject": df["Subject"].fillna(""),
            "body": df["Message"].fillna(""),
            "date": df["Date"],
        }
    )

    # ordenar por data (para facilitar splits temporais depois)
    out = out.sort_values(["date", "id"]).reset_index(drop=True)

    # Guardar em UTF-8
    DST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST_PATH, index=False)

    print(f"Dataset convertido guardado em: {DST_PATH}")
    print(out["label"].value_counts())

if __name__ == "__main__":
    main()
