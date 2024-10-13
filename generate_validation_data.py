import pandas as pd
import sys

csv_dataset_file = 'data/imdb_top_1000.csv'

VALIDATION_DATA_DIR = 'validation_data/'

csv_validation_dataset = VALIDATION_DATA_DIR + 'favorite_movies_imdb.csv'

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing the index of the row!")
        exit()
    args = sys.argv
    # Index of the row
    row_index = int(args[1])

    write_data = {
    }

    df_data = pd.read_csv(csv_dataset_file)

    # Pick a movie
    row = df_data.iloc[row_index]
    genre = row[4]
    pick_genre_list = genre.split(",")

    file = open(VALIDATION_DATA_DIR + 'picked_movie.txt', 'w')
    file.write(str(row))
    file.write("\n\n")
    file.write(str(pick_genre_list))

    print(row)
    print(pick_genre_list)
    print("")

    columns = df_data.columns

    for i in range(len(columns)):
        column = columns[i]
        write_data[column] = []

    for i, row in df_data.iterrows():
        genre = row[4]
        genre_list = genre.split(",")

        for j in range(len(genre_list)):
            if genre_list[j] in pick_genre_list:
                for k in range(len(columns)):
                    column = columns[k]
                    write_data[column].append(row[column])

    df_write_data = pd.DataFrame(write_data)
    df_write_data.to_csv(csv_validation_dataset, index=False)
