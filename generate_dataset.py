"""Generate a modified dataset from the original dataset"""

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from config import CSV_DATASET

CSV_ORIGINAL_DATASET = "original_dataset/imdb_top_1000.csv"
CSV_MOVIE_LABELS = "dataset/movie_labels.csv"


def get_label_number(df_movie_labels: DataFrame, r_genre_item: str):
    rows = df_movie_labels.loc[df_movie_labels["Genre"] == r_genre_item]
    row = rows.iloc[0]
    return row["Label_ID"]


def main():
    # 1. Read CSV files
    # Read the original dataset
    df_data = pd.read_csv(CSV_ORIGINAL_DATASET)

    # Read the movie labels data
    df_movie_labels = pd.read_csv(CSV_MOVIE_LABELS)

    # 2. Set the columns for the written data
    write_data = {}
    columns = df_data.columns

    ignore_columns = ["Poster_Link", "Star3", "Star4"]
    for i in range(len(columns)):
        column = columns[i]
        if column in ignore_columns:
            continue
        write_data[column] = []

    write_data["Movie_ID"] = []
    write_data["Label"] = []

    # 3. Set the movie id and label for each movie entry
    for i, row in df_data.iterrows():
        r_genre = row["Genre"]
        r_genre = r_genre.replace(" ", "")
        r_genre_list = r_genre.split(",")
        r_genre_item = r_genre_list[0]

        # Specify the label of the movie entry
        label_num = get_label_number(df_movie_labels, r_genre_item)
        write_data["Label"].append(label_num)

        # Specify the movie id of the movie entry
        write_data["Movie_ID"].append(i)

        # Keep the same values of the rest of the columns
        for j in range(len(columns)):
            column = columns[j]
            if column in ignore_columns:
                continue
            write_data[column].append(row[column])

    # 4. Write the dataset to a CSV file.
    df_write_data = pd.DataFrame(write_data)
    df_write_data.to_csv(CSV_DATASET, index=False)

    print(f"Written the csv file to {CSV_DATASET}")


if __name__ == "__main__":
    main()
