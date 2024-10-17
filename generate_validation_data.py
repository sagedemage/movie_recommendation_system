"""Generate a validation dataset based on the movie the user picks"""

import pandas as pd
import sys

from config import CSV_DATASET, CSV_VALIDATION_DATASET


def main():
    if len(sys.argv) < 2:
        print("Missing the index of the row!")
        exit()
    args = sys.argv
    # Index of the row
    row_index = int(args[1])

    write_data = {}

    df_data = pd.read_csv(CSV_DATASET)

    # Pick a movie
    pick_row = df_data.iloc[row_index]
    pick_genre = pick_row["Genre"]
    pick_genre = pick_genre.replace(" ", "")
    pick_genre_list = pick_genre.split(",")

    file = open("validation_data/picked_movie.txt", "w", encoding="utf-8")
    for col in df_data.columns:
        buf = f"{col}: {pick_row[col]}\n"
        file.write(buf)
    file.write("\n")
    file.write(str(pick_genre_list))
    file.close()

    print(pick_row)
    print(pick_genre_list)
    print("")

    columns = df_data.columns

    for i in range(len(columns)):
        column = columns[i]
        write_data[column] = []

    for i, row in df_data.iterrows():
        r_genre = row["Genre"]
        r_genre = r_genre.replace(" ", "")
        r_genre_list = r_genre.split(",")

        for j in range(len(r_genre_list)):
            if r_genre_list[j] in pick_genre_list:
                for k in range(len(columns)):
                    column = columns[k]
                    write_data[column].append(row[column])

    df_write_data = pd.DataFrame(write_data)

    # Remove duplicate data
    df_write_data = df_write_data.drop_duplicates()

    num_rows = df_write_data.shape[0]

    # Make sure the length of the dataset is divisible by 4
    # This is required for it to work with a batch size of 4
    # for training
    if num_rows % 4 != 0:
        rem = num_rows % 4
        df_write_data = df_write_data.drop(df_write_data.tail(rem).index)
        print(f"Number of rows: {num_rows}")
        print(f"Remainder: {rem}")

    # Write the data to a CSV file
    df_write_data.to_csv(CSV_VALIDATION_DATASET, index=False)


if __name__ == "__main__":
    main()
