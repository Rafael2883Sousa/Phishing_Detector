# tools/merge_enron_nazario.py
#
# Junta:
#   data/processed/enron_ham.csv
#   data/processed/nazario_phishing.csv
# num único data/processed/emails_full.csv
#
# Schema final: id,label,subject,body,date

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENRON_PATH = ROOT / "data" / "processed" / "enron_ham.csv"
NAZARIO_PATH = ROOT / "data" / "processed" / "nazario_phishing.csv"
DST_PATH = ROOT / "data" / "processed" / "emails_full.csv"


def main():
    if not ENRON_PATH.exists():
        raise FileNotFoundError(f"Ficheiro enron_ham.csv não encontrado: {ENRON_PATH}")
    if not NAZARIO_PATH.exists():
        raise FileNotFoundError(
            f"Ficheiro nazario_phishing.csv não encontrado: {NAZARIO_PATH}"
        )

    enron = pd.read_csv(ENRON_PATH)
    nazario = pd.read_csv(NAZARIO_PATH)

    required_cols = {"label", "subject", "body", "date"}
    missing_enron = required_cols - set(enron.columns)
    missing_naz = required_cols - set(nazario.columns)
    if missing_enron:
        raise ValueError(f"Colunas em falta em enron_ham.csv: {missing_enron}")
    if missing_naz:
        raise ValueError(f"Colunas em falta em nazario_phishing.csv: {missing_naz}")

    # Adicionar coluna de origem (opcional, útil para análise)
    enron["source"] = "enron_ham"
    nazario["source"] = "nazario_phish"

    combined = pd.concat([enron, nazario], ignore_index=True)

    # Reatribuir id sequencial
    combined = combined.reset_index(drop=True)
    combined["id"] = combined.index + 1

    # Reordenar colunas para o formato esperado
    combined = combined[["id", "label", "subject", "body", "date", "source"]]

    DST_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DST_PATH, index=False)

    print(f"Dataset combinado guardado em: {DST_PATH}")
    print(combined["label"].value_counts())
    print(combined["source"].value_counts())


if __name__ == "__main__":
    main()
