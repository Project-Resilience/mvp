"""
Extremely simple script that loads the ELUC dataset from huggingface and saves it
as a csv file for the ELUC app.
"""
from pathlib import Path

from app.constants import APP_START_YEAR
from data.eluc_data import ELUCData

def main():
    """
    Main function that loads the data and saves it.
    """
    dataset = ELUCData(APP_START_YEAR-1, APP_START_YEAR, 2022)
    test_df = dataset.test_df
    save_dir = Path("app/data")
    save_dir.mkdir(exist_ok=True)
    test_df.to_csv(save_dir / "app_data.csv")

if __name__ == "__main__":
    main()
