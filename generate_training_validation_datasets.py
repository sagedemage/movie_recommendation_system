"""
Generate the training and validation datasets
via the train-test split method
"""

import numpy as np
import pandas as pd

from config import CSV_DATASET, CSV_TRAINING_DATASET, CSV_VALIDATION_DATASET


def main():
    # 1. Read CSV files
    # Read the original dataset
    df_data = pd.read_csv(CSV_DATASET)

    # 2. Split the dataset into training and validation datasets
    # - 85 % for training
    # - 15 % for validation

    portion = [int(0.85 * len(df_data))]
    train_data, validation_data = (
        np.split(  # pylint: disable=unbalanced-tuple-unpacking
            df_data.sample(frac=1, random_state=42), portion
        )
    )

    print(len(train_data))
    print(len(validation_data))

    # 3. Write the training and validation datasets to a CSV file.
    df_write_training_dataset = pd.DataFrame(train_data)
    df_write_training_dataset.to_csv(CSV_TRAINING_DATASET, index=False)

    df_write_validation_dataset = pd.DataFrame(validation_data)
    df_write_validation_dataset.to_csv(CSV_VALIDATION_DATASET, index=False)

    print(f"Written the training dataset csv file to {CSV_TRAINING_DATASET}")
    print(
        f"Written the validation dataset csv file to {CSV_VALIDATION_DATASET}"
    )


if __name__ == "__main__":
    main()
