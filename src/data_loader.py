from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "placement_data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)