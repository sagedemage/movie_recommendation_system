"""
Generate a personalized dataset based on the
movie the user picks via a GUI interface
"""

import tkinter as tk
from tkinter import ttk
from tkinter.constants import CENTER, RIGHT, Y, NO, LEFT, BOTH, BOTTOM, X

from config import CSV_DATASET, CSV_PERSONALIZED_DATASET, PICKED_MOVIE_TEXT_FILE
import pandas as pd

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_X_POS = 50
WINDOW_Y_POS = 85


def main():
    df_data = pd.read_csv(CSV_DATASET)
    root = tk.Tk()
    root.title("Generate Personalized Data")
    root.geometry(
        f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{WINDOW_X_POS}+{WINDOW_Y_POS}"
    )
    frame = tk.Frame(root)
    w = tk.Label(
        root, text="Pick your favorite movie. Press enter to choose the movie."
    )
    w.pack()
    frame.pack()

    y_scroll_bar = tk.Scrollbar(frame)
    y_scroll_bar.pack(side=RIGHT, fill=Y)

    x_scroll_bar = tk.Scrollbar(frame, orient="horizontal")
    x_scroll_bar.pack(side=BOTTOM, fill=X)

    table = ttk.Treeview(
        frame,
        yscrollcommand=y_scroll_bar.set,
        xscrollcommand=x_scroll_bar.set,
        selectmode="browse",
        height=22,
    )
    columns = df_data.columns
    table["columns"] = (
        columns[0],
        columns[1],
        columns[2],
        columns[4],
        columns[5],
        columns[6],
        columns[7],
        columns[9],
    )

    column_width = 100
    table.column("#0", width=0, stretch=NO)
    table.column(columns[0], anchor=CENTER, width=column_width)
    table.column(columns[1], anchor=CENTER, width=325)
    table.column(columns[2], anchor=CENTER, width=column_width)
    table.column(columns[4], anchor=CENTER, width=column_width)
    table.column(columns[5], anchor=CENTER, width=220)
    table.column(columns[6], anchor=CENTER, width=column_width)
    table.column(columns[7], anchor=CENTER, width=325)
    table.column(columns[9], anchor=CENTER, width=160)

    table.heading("#0", text="", anchor=CENTER)
    table.heading(columns[0], text="Movie ID", anchor=CENTER)
    table.heading(columns[1], text="Series Title", anchor=CENTER)
    table.heading(columns[2], text="Released Year", anchor=CENTER)
    table.heading(columns[4], text="Runtime", anchor=CENTER)
    table.heading(columns[5], text="Genre", anchor=CENTER)
    table.heading(columns[6], text="IMDB Rating", anchor=CENTER)
    table.heading(columns[7], text="Overview", anchor=CENTER)
    table.heading(columns[9], text="Director", anchor=CENTER)

    for i in range(len(df_data)):
        movie_id = df_data.iloc[i]["Movie_ID"]
        series_title = df_data.iloc[i]["Series_Title"]
        released_year = df_data.iloc[i]["Released_Year"]
        runtime = df_data.iloc[i]["Runtime"]
        genre = df_data.iloc[i]["Genre"]
        imdb_rating = df_data.iloc[i]["IMDB_Rating"]
        overview = df_data.iloc[i]["Overview"]
        director = df_data.iloc[i]["Director"]
        r_values = (
            movie_id,
            series_title,
            released_year,
            runtime,
            genre,
            imdb_rating,
            overview,
            director,
        )
        table.insert(parent="", index="end", iid=i, text="", values=r_values)

    def item_selected(event):  # pylint: disable=unused-argument
        # 1. Retrieve the information of the picked movie
        # Pick a movie
        selected_index = table.focus()
        selected_item = table.item(selected_index)
        item_values = selected_item["values"]
        pick_genre = item_values[4]
        pick_genre_list = pick_genre.split(", ")
        print(f"Row: {item_values}")
        print(f"Genre: {pick_genre_list}")

        # Log the information of the movie to remember what movie the user chose
        file = open(PICKED_MOVIE_TEXT_FILE, "w", encoding="utf-8")

        pick_movie_id = item_values[0]
        pick_row = df_data.iloc[pick_movie_id]
        index = 0
        for col in columns:
            buf = f"{col}: {pick_row[col]}\n"
            file.write(buf)
            index += 1
        file.write("\n")
        file.write(str(pick_genre_list))
        file.close()

        # 2. Add movies where one of their genres is in the picked
        # movie's genre. This is to use as training data for
        # recommendation of movies.
        write_data = {}

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

        # 4. Write the data to a CSV file.
        df_write_data.to_csv(CSV_PERSONALIZED_DATASET, index=False)

        print(
            f"Written the personalized dataset csv file "
            f"to {CSV_PERSONALIZED_DATASET}"
        )
        print(
            f"Written the picked movie information to {PICKED_MOVIE_TEXT_FILE}"
        )
        print("")

    def item_overview(event):  # pylint: disable=unused-argument
        # 1. Retrieve the information of the picked movie
        # Pick a movie
        selected_index = table.focus()
        selected_item = table.item(selected_index)
        item_values = selected_item["values"]

        # 2. Print the overview of the selected movie
        if len(item_values) == 8:
            view_overview = item_values[6]
            print(f"Overview: {view_overview}")
            print("")

    table.bind("<Return>", item_selected)
    table.bind("<Control-Button>", item_overview)
    y_scroll_bar.config(command=table.yview)
    x_scroll_bar.config(command=table.xview)
    table.pack(side=LEFT, fill=BOTH)
    root.mainloop()


if __name__ == "__main__":
    main()
