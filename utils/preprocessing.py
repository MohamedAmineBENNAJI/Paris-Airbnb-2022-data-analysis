"""This module contains the utility functions used in preprocessing"""
from typing import List

import pandas as pd


def preprocess(
    data: pd.DataFrame, columns_of_interest: List[str] = ["host_neighbourhood"]
) -> pd.DataFrame:
    """This utility function is used to preprocess the dataframes
        and remove the missing values from the columns of interest.

    Arguments:
        data: The input dataframe.
        columns_of_interest: A list containing the used columns to preprocess.

    Returns:
        data: Preprocessed dataframe.
    """
    data = data.drop_duplicates()
    data = data.dropna(axis=1, how="all")
    data = data.dropna(subset=columns_of_interest)
    data.rename(
        columns={
            "id": "listing_id",
            "calendar_last_scraped": "date",
            "has_availability": "available",
        },
        inplace=True,
    )

    return data


def format_prices(
    data: pd.DataFrame, columns: List[str] = ["price", "adjusted_price"]
) -> pd.DataFrame:
    """This utility function is used to format the prices and make them numeric.

    Arguments:
        data: The input dataframe.
        columns: A list of price columns to format.

    Returns:
        data: The output dataframe with formatted price columns."""
    for col in columns:

        data[col] = data[col].str.replace("$", "", regex=True)
        data[col] = data[col].str.replace(",", "", regex=True)
        data[col] = pd.to_numeric(data[col])

    return data


def format_availability(data: pd.DataFrame, column: str = "available") -> pd.DataFrame:
    """This utility function is used to format the availability column
        and make it boolean.

    Arguments:
        data: The input dataframe.
        column: The availability column name.

    Returns:
        data: The output dataframe with formatted availability column."""
    data.loc[(data[column] == "t"), column] = True
    data.loc[(data[column] == "f"), column] = False

    return data
