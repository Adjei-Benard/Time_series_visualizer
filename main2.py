

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Output directory
output_dir = Path(__file__).parent / "output1"
os.makedirs(output_dir, exist_ok=True)

# 1. Load data

data_file = Path("fcc-forum-pageviews.csv")

if data_file.exists():
    df = pd.read_csv(data_file, parse_dates=["date"], index_col="date")
else:
    # Synthetic fallback so the script can run standalone
    rng = pd.date_range("2016-05-09", "2019-12-03", freq="D")
    np.random.seed(42)
    views = (
        20000
        + np.random.normal(0, 2000, len(rng)).cumsum()
        + np.sin(np.linspace(0, 40 * np.pi, len(rng))) * 3000
    ).astype(int)
    df = pd.DataFrame({"value": views}, index=rng)
    df.index.name = "date"

# Ensure correct column name
df.rename(columns={df.columns[0]: "value"}, inplace=True)

# 2. Clean data: remove top and bottom 2.5%
lower = df["value"].quantile(0.025)
upper = df["value"].quantile(0.975)
df_clean = df[(df["value"] >= lower) & (df["value"] <= upper)]

df_clean.to_csv(output_dir / "clean_page_views.csv")

# 3. Line plot

def draw_line_plot(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data.index, data["value"], color="tab:red", linewidth=1)
    ax.set_title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019")
    ax.set_xlabel("Date")
    ax.set_ylabel("Page Views")
    fig.tight_layout()
    fig.savefig(output_dir / "line_plot.png")
    return fig

# 4. Bar plot

def draw_bar_plot(data: pd.DataFrame) -> plt.Figure:
    df_bar = (
        data.copy()
        .assign(year=lambda d: d.index.year, month=lambda d: d.index.month_name())
        .groupby(["year", "month"])
        .mean()
        .reset_index()
    )
    month_order = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
    ]
    df_bar["month"] = pd.Categorical(df_bar["month"], categories=month_order, ordered=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df_bar,
        x="year",
        y="value",
        hue="month",
        hue_order=month_order,
        ax=ax,
    )
    ax.set_xlabel("Years")
    ax.set_ylabel("Average Page Views")
    ax.set_title("Months")
    ax.legend(title="Months", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "bar_plot.png")
    return fig

# 5. Box plots

def draw_box_plot(data: pd.DataFrame) -> plt.Figure:
    df_box = data.copy()
    df_box["year"] = df_box.index.year
    df_box["month"] = df_box.index.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df_box["month"] = pd.Categorical(df_box["month"], categories=month_order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(x="year", y="value", data=df_box, ax=axes[0])
    axes[0].set_title("Year-wise Box Plot (Trend)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Page Views")
    sns.boxplot(x="month", y="value", data=df_box, ax=axes[1])
    axes[1].set_title("Month-wise Box Plot (Seasonality)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Page Views")
    fig.tight_layout()
    fig.savefig(output_dir / "box_plot.png")
    return fig

# Run visualizations when this script is executed
if __name__ == "__main__":
    draw_line_plot(df_clean)
    draw_bar_plot(df_clean)
    draw_box_plot(df_clean)
    print("Images saved in output1: line_plot.png, bar_plot.png, box_plot.png\nClean data saved: clean_page_views.csv")
