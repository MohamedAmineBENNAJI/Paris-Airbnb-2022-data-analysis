"""This module contains the utility functions used in preprocessing"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_neighbourhoods(
    data: pd.DataFrame,
    address_column: str = "host_neighbourhood",
    price_column: str = "adjusted_price",
    number_of_samples: int = 10,
    most_expensive: bool = True,
    is_available: bool = True,
) -> None:
    """This utility function is used to plot the most expensive and cheapest neighbourhoods according to certain conditions.

    Arguments:
        data: The input dataframe.
        address_column: The column containing the host's neighbourhood address.
        price_column: The column indicating the final prices for the listing.
        number_of_samples: The number of samples to plot.
        most_expensive: A boolean indicating whether to plot the most expensive or the cheapest listings.
        is_available: A boolean indiating the availability of the listings.
    """
    availability = {True: "available", False: "booked"}
    if most_expensive:

        title = f"Top {number_of_samples} most expensive {availability[is_available]} listings in Paris"
    else:
        title = f"Top {number_of_samples} cheapest {availability[is_available]} listings in Paris"

    data = data.loc[data.available == is_available]
    neighbourhoods = (
        data.groupby(address_column)
        .mean()[price_column]
        .sort_values(ascending=not (most_expensive))
    )
    neighbourhoods_names = neighbourhoods[:number_of_samples].keys()
    prices = neighbourhoods[:number_of_samples].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
    palette_color = sns.color_palette("bright")
    fig.suptitle(title, fontweight="bold", fontsize=30, y=0.95)

    ax1.bar(neighbourhoods_names, prices, color=palette_color)
    ax1.set_ylabel("Price($)", fontsize=20)
    ax1.set_xlabel("Neighbourhoods", fontsize=20)
    ax1.tick_params(axis="y", labelsize=15)

    # Rotation of the bars names
    ax1.tick_params(axis="x", labelrotation=90, labelsize=15)

    # plotting data on chart
    ax2.pie(
        prices,
        labels=neighbourhoods_names,
        colors=palette_color,
        autopct="%.0f%%",
        textprops={"fontsize": 15},
    )
    ax2.legend(bbox_to_anchor=(0.5, 1.5), prop={"size": 15})
    # displaying chart
    plt.show()
