from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def main():
    print("Prooflink skeleton is wired up.")
    # Example placeholder – update once your real files exist
    # df = load_csv("payroll_sample.csv")
    # print(df.head())


if __name__ == "__main__":
    main()
