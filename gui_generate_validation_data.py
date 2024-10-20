"""
Generate a validation dataset based on the
movie the user picks via a GUI interface
"""

import tkinter as tk
from tkinter import ttk
from tkinter import Frame, Scrollbar
from tkinter.constants import CENTER, RIGHT, Y, NO, LEFT, BOTH, BOTTOM, X

from config import CSV_DATASET, CSV_VALIDATION_DATASET
import pandas as pd

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_X_POS = 50
WINDOW_Y_POS = 85


def main():
    df_data = pd.read_csv(CSV_DATASET)
    root = tk.Tk()
    root.title("Test window")
    root.geometry(
        f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{WINDOW_X_POS}+{WINDOW_Y_POS}"
    )
    frame = Frame(root)
    w = tk.Label(
        root, text="Pick your favorite movie. Press enter to choose the movie."
    )
    w.pack()
    frame.pack()

    y_scroll_bar = Scrollbar(frame)
    y_scroll_bar.pack(side=RIGHT, fill=Y)

    x_scroll_bar = Scrollbar(frame, orient="horizontal")
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
        columns[9],
    )

    column_width = 160
    table.column("#0", width=0, stretch=NO)
    table.column(columns[0], anchor=CENTER, width=100)
    table.column(columns[1], anchor=CENTER, width=325)
    table.column(columns[2], anchor=CENTER, width=100)
    table.column(columns[4], anchor=CENTER, width=100)
    table.column(columns[5], anchor=CENTER, width=column_width)
    table.column(columns[6], anchor=CENTER, width=100)
    table.column(columns[9], anchor=CENTER, width=column_width)

    table.heading("#0", text="", anchor=CENTER)
    table.heading(columns[0], text="Movie ID", anchor=CENTER)
    table.heading(columns[1], text="Series Titles", anchor=CENTER)
    table.heading(columns[2], text="Released Year", anchor=CENTER)
    table.heading(columns[4], text="Runtime", anchor=CENTER)
    table.heading(columns[5], text="Genre", anchor=CENTER)
    table.heading(columns[6], text="IMDB Rating", anchor=CENTER)
    table.heading(columns[9], text="Director", anchor=CENTER)

    for i in range(len(df_data)):
        movie_id = df_data.iloc[i]["Movie_ID"]
        series_title = df_data.iloc[i]["Series_Title"]
        released_year = df_data.iloc[i]["Released_Year"]
        runtime = df_data.iloc[i]["Runtime"]
        genre = df_data.iloc[i]["Genre"]
        imdb_rating = df_data.iloc[i]["IMDB_Rating"]
        director = df_data.iloc[i]["Director"]
        r_values = (
            movie_id,
            series_title,
            released_year,
            runtime,
            genre,
            imdb_rating,
            director,
        )
        table.insert(parent="", index="end", iid=i, text="", values=r_values)

    def item_selected(event):
        selected_index = table.focus()
        selected_item = table.item(selected_index)
        item_values = selected_item["values"]
        pick_genre = item_values[4]
        pick_genre_list = pick_genre.split(", ")
        print(f"Row: {item_values}")
        print(f"Genre: {pick_genre_list}")

    table.bind("<Return>", item_selected)
    y_scroll_bar.config(command=table.yview)
    x_scroll_bar.config(command=table.xview)
    table.pack(side=LEFT, fill=BOTH)
    root.mainloop()


if __name__ == "__main__":
    main()
