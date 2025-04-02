import argparse
import os
import sqlite3
import textwrap
from datetime import datetime
from time import sleep

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LogNorm

from utils.helpers import DB_PATH, get_settings


def get_data(now=None):
    """Retrieves detection data from the database for a given date."""
    try:
        conn = sqlite3.connect(DB_PATH)
        if now is None:
            now = datetime.now()
        df = pd.read_sql_query(
            f"SELECT * from detections WHERE Date = DATE('{now.strftime('%Y-%m-%d')}')",
            conn,
        )

        # Convert Date and Time Fields to Panda's format
        df["Date"] = pd.to_datetime(df["Date"])
        df["Time"] = pd.to_datetime(df["Time"], unit="ns")

        # Add round hours to dataframe
        df["Hour of Day"] = [r.hour for r in df.Time]

        return df, now
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return pd.DataFrame(), now  # Return empty DataFrame on error
    finally:
        if conn:
            conn.close()


def show_values_on_bars(ax, label):
    """Displays values on bars in a countplot."""
    conf = get_settings()

    for i, p in enumerate(ax.patches):
        x = p.get_x() + p.get_width() * 0.9
        y = p.get_y() + p.get_height() / 2
        value = "{:n}".format(p.get_width())
        bbox = {
            "facecolor": "#a5f5b3" if conf["COLOR_SCHEME"] != "dark" else "#005225",
            "edgecolor": "none",
            "pad": 1.0,
        }
        if conf["COLOR_SCHEME"] == "dark":
            color = "#a5f5b3"
        else:
            color = "#00210b"

        ax.text(x, y, value, bbox=bbox, ha="center", va="center", size=9, color=color)


def wrap_width(txt):
    """Calculates wrap width for text."""
    w = 16
    for c in txt:
        if c in ["M", "m", "W", "w"]:
            w -= 0.33
        if c in ["I", "i", "j", "l"]:
            w += 0.33
    return round(w)


def create_heatmap(df, freq_order, now, conf, ax):
    """Creates the heatmap subplot."""
    heat = pd.crosstab(df["Com_Name"], df["Hour of Day"])
    heat.index = pd.CategoricalIndex(heat.index, categories=freq_order)
    heat.sort_index(level=0, inplace=True)

    hours_in_day = pd.Series(data=range(0, 24))
    heat_frame = pd.DataFrame(data=0, index=heat.index, columns=hours_in_day)
    heat = (heat + heat_frame).fillna(0)
    heat[heat == 0] = np.nan

    heatmap_plot = sns.heatmap(
        heat,
        norm=LogNorm(),
        annot=True,
        annot_kws={"fontsize": 10},
        fmt="g",
        cmap=conf["PALETTE"],
        square=True,
        cbar=False,
        linewidths=0.5,
        linecolor="Grey",
        ax=ax,
    )

    heatmap_plot.set(ylabel=None)
    heatmap_plot.set(xlabel=None)

    # Set color and weight of tick label for current hour
    for label in heatmap_plot.get_xticklabels():
        if int(label.get_text()) == now.hour:
            if conf["COLOR_SCHEME"] == "dark":
                label.set_color("white")
            else:
                label.set_color("yellow")

    heatmap_plot.set_xticklabels(heatmap_plot.get_xticklabels(), rotation=0, size=10)

    # Set heatmap border
    for _, spine in heatmap_plot.spines.items():
        spine.set_visible(True)
    heatmap_plot.set(ylabel=None)
    heatmap_plot.set(xlabel="Hour of Day")

    return heatmap_plot


def create_countplot(df, freq_order, confmax, conf, ax):
    """Creates the countplot subplot."""
    countplot_plot = sns.countplot(
        y="Com_Name",
        hue="Com_Name",
        legend=False,
        data=df,
        palette=dict(zip(confmax.index, conf["COLORS"])),
        order=freq_order,
        ax=ax,
        edgecolor="lightgrey",
    )

    show_values_on_bars(ax, confmax)

    countplot_plot.set(ylabel=None)
    countplot_plot.set(xlabel="Detections")
    countplot_plot.set_yticklabels([])  # Remove countplot y-axis labels

    return countplot_plot


def apply_formatting(fig, heatmap_plot, count_plot, plot_type, readings, now, conf, height):
    """Applies formatting to the plots (titles, labels, etc.)."""

    # Get the y-axis ticks and labels from the heatmap
    heatmap_yticks = heatmap_plot.get_yticks()

    # Apply the same y-axis ticks and labels to the countplot
    count_plot.set_ylim(heatmap_plot.get_ylim())
    count_plot.set_yticks(heatmap_yticks)

    # Set combined plot layout and titles
    y = 1 - 8 / (height * 100)
    title_color = "#e5e2e0" if conf["COLOR_SCHEME"] == "dark" else "#1c1b1b"
    plt.suptitle(
        f"{plot_type} {readings} Last Updated: {now.strftime('%Y-%m-%d %H:%M')}",
        y=y,
        color=title_color,
    )
    fig.tight_layout()
    top = 1 - 40 / (height * 100)
    fig.subplots_adjust(left=0.15, right=0.9, top=top, wspace=0)  # increase left margin.


def create_combined_plot(df_plt_today, now, conf, is_top=None):
    """Creates the combined heatmap and countplot."""
    if is_top is not None:
        readings = 10
        if is_top:
            plt_selection_today = df_plt_today["Com_Name"].value_counts()[:readings]
        else:
            plt_selection_today = df_plt_today["Com_Name"].value_counts()[-readings:]
    else:
        plt_selection_today = df_plt_today["Com_Name"].value_counts()
        readings = len(df_plt_today["Com_Name"].value_counts())

    df_plt_selection_today = df_plt_today[
        df_plt_today.Com_Name.isin(plt_selection_today.index)
    ]

    # Set up plot axes and titles
    height = (max(readings / 5, 0) + 1.06) * 1.2  # increase the height by 20%
    if conf["COLOR_SCHEME"] == "dark":
        facecolor = "none"
    else:
        facecolor = "none"

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(10, height),
        gridspec_kw=dict(width_ratios=[6, 3]),
        facecolor=facecolor,
    )

    label_color = "#a5f5b3" if conf["COLOR_SCHEME"] == "dark" else "#00210b"

    for ax in axs:
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        ax.tick_params(axis="x", colors=label_color)
        ax.tick_params(axis="y", colors=label_color)

    # generate y-axis order for all figures based on frequency
    freq_order = df_plt_selection_today["Com_Name"].value_counts().index

    # make color for max confidence --> this groups by name and calculates max conf
    confmax = df_plt_selection_today.groupby("Com_Name")["Confidence"].max()
    # reorder confmax to detection frequency order
    confmax = confmax.reindex(freq_order)

    # norm values for color palette
    norm = plt.Normalize(confmax.values.min(), confmax.values.max())
    if is_top or is_top is None:
        # Set Palette for graphics
        if conf["COLOR_SCHEME"] == "dark":
            conf["PALETTE"] = "magma"
            conf["COLORS"] = plt.cm.magma(norm(confmax)).tolist()
        else:
            conf["PALETTE"] = "viridis"
            conf["COLORS"] = plt.cm.viridis(norm(confmax)).tolist()
        if is_top:
            plot_type = "Top"
        else:
            plot_type = "All"
        name = "Combo"
    else:
        # Set Palette for graphics
        conf["PALETTE"] = "Reds"
        conf["COLORS"] = plt.cm.Reds(norm(confmax)).tolist()
        plot_type = "Bottom"
        name = "Combo2"

    try:
        # Create the plots
        heatmap_plot = create_heatmap(df_plt_selection_today, freq_order, now, conf, axs[0])
        countplot_plot = create_countplot(
            df_plt_selection_today, conf_order, confmax, conf, axs[1]
        )
        # Apply formatting
        apply_formatting(fig, heatmap_plot, countplot_plot, plot_type, readings, now, conf, height)

        # Save combined plot
        save_name = os.path.expanduser(
            f"~/BirdSongs/Extracted/Charts/{name}-{now.strftime('%Y-%m-%d')}.png"
        )
        plt.savefig(save_name, transparent=True)
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error creating plot: {e}")


def load_fonts():
    """Loads fonts from the specified directory."""
    conf = get_settings()
    # Add every font at the specified location
    font_dir = [os.path.expanduser("~/BirdNET-Pi/homepage/static")]
    for font in font_manager.findSystemFonts(font_dir, fontext="ttf"):
        font_manager.fontManager.addfont(font)
    # Set font family globally
    if conf["DATABASE_LANG"] in ["ja", "zh"]:
        rcParams["font.family"] = "Noto Sans JP"
    elif conf["DATABASE_LANG"] == "th":
        rcParams["font.family"] = "Noto Sans Thai"
    else:
        rcParams["font.family"] = "Roboto Flex"


def main(daemon, sleep_m):
    load_fonts()
    last_run = None
    while True:
        now = datetime.now()
        # now = datetime.strptime('2023-12-13T23:59:59', "%Y-%m-%dT%H:%M:%S")
        # now = datetime.strptime('2024-01-02T23:59:59', "%Y-%m-%dT%H:%M:%S")
        # now = datetime.strptime('2024-02-26T23:59:59', "%Y-%m-%dT%H:%M:%S")
        # now = datetime.strptime('2024-04-03T23:59:59', "%Y-%m-%dT%H:%M:%S")
        # now = datetime.strptime('2024-04-07T23:59:59', "%Y-%m-%dT%H:%M:%S")
        if last_run and now.day != last_run.day:
            print("getting yesterday's dataset")
            yesterday = last_run.replace(hour=23, minute=59)
            data, time = get_data(yesterday)
        else:
            data, time = get_data(now)
        if not data.empty:
            conf = get_settings()
            create_combined_plot(data, time, conf)
        else:
            print("empty dataset")
        if daemon:
            last_run = now
            sleep(60 * sleep_m)
        else:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument(
        "--sleep", default=2, type=int, help="Time between runs (minutes)"
    )
    args = parser.parse_args()
    main(args.daemon, args.sleep)