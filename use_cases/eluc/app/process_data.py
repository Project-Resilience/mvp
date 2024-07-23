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
    # Subsets the dataset so train_df is from start_year-1 to test year which we discard.
    # Then we take the app data as the test def which is from the app start year to the end of the dataset.
    dataset = ELUCData.from_hf(start_year=APP_START_YEAR-1, test_year=APP_START_YEAR)
    test_df = dataset.test_df
    save_dir = Path("app/data")
    save_dir.mkdir(exist_ok=True)
    test_df.to_csv(save_dir / "app_data.csv")


if __name__ == "__main__":
    main()
