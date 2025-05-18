# scripts/clean.py
"""
This script is responsible for loading raw fake and real news datasets
(assumed to be in CSV format), cleaning the 'title' text data, adding
labels (0 for fake, 1 for real), combining the datasets, and saving the
processed data to a new CSV file.

The cleaning steps include converting text to lowercase, removing URLs,
removing punctuation and numbers, and stripping extra whitespace.
"""

import pandas as pd
import os
import re

# Input paths
FAKE_PATH = "../project_data/fake.csv"
REAL_PATH = "../project_data/real.csv"

# Output path
OUT_PATH = "../project_data/processed_titles.csv"


def clean_text(text):
    """
    Cleans a single string of text by performing several operations.

    Cleaning steps include:
    - Handling non-string inputs by returning an empty string.
    - Converting the text to lowercase.
    - Removing URLs (http/https links).
    - Removing any characters that are not English letters or whitespace.
    - Reducing multiple spaces to a single space and stripping leading/trailing whitespace.

    Args:
        text (str or any): The input text string to be cleaned.

    Returns:
        str: The cleaned text string. Returns an empty string if the input
             is not a string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def main():
    """
    Main function to execute the data cleaning and preparation pipeline.

    Loads the fake and real news CSVs, adds labels, selects and cleans
    the 'title' column, combines the datasets, shuffles the combined data,
    and saves the result to a new CSV file.
    """
    print(f"Loading data from {FAKE_PATH} and {REAL_PATH}...")
    # Load CSVs
    try:
        fake_df = pd.read_csv(FAKE_PATH)
        real_df = pd.read_csv(REAL_PATH)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading raw data files: {e}")
        print(
            f"Please ensure '{os.path.basename(FAKE_PATH)}' and '{os.path.basename(REAL_PATH)}' exist in '{os.path.dirname(FAKE_PATH)}'."
        )
        return  # Exit if files aren't found

    # Add labels
    fake_df["label"] = 0  # 0 for fake
    real_df["label"] = 1  # 1 for real
    print("Added labels to datasets.")

    # Keep only title and label
    fake_df = fake_df[["title", "label"]]
    real_df = real_df[["title", "label"]]
    print("Selected 'title' and 'label' columns.")

    # Drop missing titles
    initial_fake_rows = len(fake_df)
    initial_real_rows = len(real_df)
    fake_df.dropna(subset=["title"], inplace=True)
    real_df.dropna(subset=["title"], inplace=True)
    print(
        f"Dropped {initial_fake_rows - len(fake_df)} rows with missing titles from fake data."
    )
    print(
        f"Dropped {initial_real_rows - len(real_df)} rows with missing titles from real data."
    )

    # Combine and clean
    print("Combining datasets and cleaning titles...")
    combined_df = pd.concat([fake_df, real_df], ignore_index=True)
    combined_df["title"] = combined_df["title"].apply(clean_text)
    print("Cleaning complete.")

    # Shuffle
    print("Shuffling combined data...")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Data shuffled.")

    # Save
    output_dir = os.path.dirname(OUT_PATH)
    if (
        output_dir
    ):  # Check if output_dir is not empty string (i.e., not current directory)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory '{output_dir}' exists.")

    combined_df.to_csv(OUT_PATH, index=False)

    print(f"Saved cleaned titles to {OUT_PATH} ({len(combined_df)} samples)")


if __name__ == "__main__":
    main()
