"""One-time script to generate the bundled 10K-row sample.

Usage:
    1. Download creditcard.csv from Kaggle:
       https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    2. Place it in data/creditcard.csv
    3. Run: python data/generate_sample.py
    4. Output: data/creditcard_sample.csv (10K rows, stratified)
"""

import pandas as pd

INPUT = "data/creditcard.csv"
OUTPUT = "data/creditcard_sample.csv"
SAMPLE_SIZE = 10000


def main():
    df = pd.read_csv(INPUT)
    print(f"Full dataset: {len(df)} rows, {int(df['Class'].sum())} frauds")

    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    # Keep all frauds (492), sample rest to reach 10K
    n_normal = SAMPLE_SIZE - len(fraud)
    normal_sample = normal.sample(n=n_normal, random_state=42)

    sample = pd.concat([normal_sample, fraud]).sample(frac=1, random_state=42)
    sample.to_csv(OUTPUT, index=False)
    print(f"Sample: {len(sample)} rows, {int(sample['Class'].sum())} frauds ({sample['Class'].mean():.2%})")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
