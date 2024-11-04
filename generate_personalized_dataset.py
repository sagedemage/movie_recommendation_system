"""Generate a personalized dataset based on the movie the user picks"""

import pandas as pd
import sys

from config import CSV_DATASET, CSV_PERSONALIZED_DATASET, PICKED_MOVIE_TEXT_FILE


def main():
    # 1. Retrieve the information of the picked movie
    if len(sys.argv) < 2:
        print("Missing the index of the row!")
        exit()
    args = sys.argv

    df_data = pd.read_csv(CSV_DATASET)

    # Pick a movie
    row_index = int(args[1])
    pick_row = df_data.iloc[row_index]
    pick_genre = pick_row["Genre"]
    pick_genre_list = pick_genre.split(", ")

    # Log the information of the movie to remember what movie the user chose
    file = open(PICKED_MOVIE_TEXT_FILE, "w", encoding="utf-8")
    for col in df_data.columns:
        buf = f"{col}: {pick_row[col]}\n"
        file.write(buf)
    file.write("\n")
    file.write(str(pick_genre_list))
    file.close()

    print(pick_row)
    print(pick_genre_list)
    print("")

    # 2. Add movies where one of their genres is in the picked movie's genre.
    # This is to use as training data for recommendation of movies.
    write_data = {}
    columns = df_data.columns

    for i in range(len(columns)):
        column = columns[i]
        write_data[column] = []

    # Filter the data based on the genre of the picked movie
    for i, row in df_data.iterrows():
        r_genre = row["Genre"]
        r_genre_list = r_genre.split(", ")

        for j in range(len(r_genre_list)):
            if r_genre_list[j] in pick_genre_list:
                for k in range(len(columns)):
                    column = columns[k]
                    write_data[column].append(row[column])

    # 3. Prepare the written data before writing it to a CSV file.
    df_write_data = pd.DataFrame(write_data)
    df_write_data = df_write_data.drop_duplicates()

    num_rows = df_write_data.shape[0]

    # Make sure the length of the dataset is divisible by 4
    # This is required for it to work with a batch size of 4
    # for training
    if num_rows % 4 != 0:
        rem = num_rows % 4
        df_write_data = df_write_data.drop(df_write_data.tail(rem).index)

    # 4. Write the data to a CSV file.
    df_write_data.to_csv(CSV_PERSONALIZED_DATASET, index=False)

    print(f"Written the csv file to {CSV_PERSONALIZED_DATASET}")
    print(f"Written the picked movie information to {PICKED_MOVIE_TEXT_FILE}")


if __name__ == "__main__":
    main()
