"""This module contains the utility functions used in preprocessing"""
from typing import List

import pandas as pd
import re

from collections import Counter


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


def create_new_features(data: pd.DataFrame) -> pd.DataFrame:
    """This utility function allows us to create some new features from our categorical columns.

    Arguments:
        data: The input dataframe.

    Returns:
        data: The output dataframe.

    """
    data["shared_bath"] = data["bathrooms_text"].str.findall(
        "shared", flags=re.IGNORECASE
    )
    data["private_bath"] = data["bathrooms_text"].str.findall(
        "private", flags=re.IGNORECASE
    )
    data.shared_bath = data.shared_bath.explode().fillna(-1)
    data.private_bath = data.private_bath.explode().fillna(-1)

    data.loc[data.shared_bath != -1, "shared_bath"] = 1
    data.loc[data.private_bath != -1, "private_bath"] = 1
    data[["shared_bath", "private_bath"]] = data[
        ["shared_bath", "private_bath"]
    ].astype(int)

    data["host_in_france"] = (
        data["host_location"].str.contains("france|fr", case=False).astype(int)
    )

    return data


def clean_bathrooms_data(
    data: pd.DataFrame, column: str = "bathrooms_text"
) -> pd.DataFrame:
    """This utility function allows us to clean the bathrooms_text column
    and generate ne features from it.

    Arguments:
        data: The input dataframe.
        column: The column of interest.

    Returns:
        data: The output dataframe preprocessed.
    """
    data = create_new_features(data)

    data["bathrooms_text"] = data.bathrooms_text.str.replace("bath|Bath", "")
    data["bathrooms_text"] = data.bathrooms_text.str.replace("shared|Shared", "")

    data["bathrooms_text"] = data.bathrooms_text.str.replace("s", "")

    data["bathrooms_text"] = data.bathrooms_text.str.replace("Private|private", "")
    data["bathrooms_text"] = data.bathrooms_text.str.replace("half|Half", "0.5")
    data["bathrooms_text"] = data.bathrooms_text.str.replace("-", "")

    data["bathrooms_text"] = data["bathrooms_text"].astype("float16")

    return data


def format_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """This utility function allows us to format some categorical columns for data cleaning.

    Arguments:
        data: The input dataframe.

    Returns:
        data: The output dataframe with formatted columns.
    """

    data["host_response_rate"] = (
        data["host_response_rate"].astype("str").str.rstrip("%").astype(float)
    )
    data["host_acceptance_rate"] = (
        data["host_acceptance_rate"].astype("str").str.rstrip("%").astype(float)
    )
    data["available"] = (
        data.available.str.replace("t", "1").replace("f", "0").astype(int)
    )
    data["host_is_superhost"] = (
        data.host_is_superhost.str.replace("t", "1").replace("f", "0").astype(int)
    )
    data["host_has_profile_pic"] = (
        data.host_has_profile_pic.str.replace("t", "1").replace("f", "0").astype(int)
    )
    data["host_identity_verified"] = (
        data.host_identity_verified.str.replace("t", "1").replace("f", "0").astype(int)
    )
    data["instant_bookable"] = (
        data.instant_bookable.str.replace("t", "1").replace("f", "0").astype(int)
    )
    data.loc[
        data["host_neighbourhood"] == data["neighbourhood"], "host_neighbourhood"
    ] = 1
    data.loc[data["host_neighbourhood"] != 1, "host_neighbourhood"] = 0
    data.rename(
        columns={"host_neighbourhood": "airbnb_in_host_neighbourhood"}, inplace=True
    )

    data["airbnb_in_host_neighbourhood"] = data["airbnb_in_host_neighbourhood"].astype(
        int
    )
    data["host_since"] = pd.to_datetime(data["host_since"]).astype(int)
    data["last_review"] = pd.to_datetime(data["last_review"]).astype(int)
    data["first_review"] = pd.to_datetime(data["first_review"]).astype(int)
    data["last_scraped"] = pd.to_datetime(data["last_scraped"]).astype(int)
    data["date"] = pd.to_datetime(data["date"]).astype(int)
    data.drop(
        columns=[
            "neighbourhood",
            "host_location",
            "neighbourhood_cleansed",
            "property_type",
        ],
        inplace=True,
    )

    return data


def format_amenities(data: pd.DataFrame) -> pd.DataFrame:
    """This utility function is used to format amenities column.

    Arguments:
        data: The input dataframe.

    Returns:
        data: The output dataframe with formatted amenities."""
    # make a dictionary from the amenities
    amenities_counter = Counter()
    data["amenities"].astype("str").str.strip("[]").str.replace('"', "").str.split(
        ","
    ).apply(amenities_counter.update)

    # for the purpose of the project I'll take only the most 30 common amenities
    for item, _ in amenities_counter.most_common(30):
        col_name = "amenity_" + item.replace(" ", "_")
        data[col_name] = data["amenities"].astype("str").apply(lambda x: int(item in x))
    data.drop(columns=["amenities"], axis=1, inplace=True)

    return data


def format_host_verifications(data: pd.DataFrame) -> pd.DataFrame:
    """This utility function is used to format host_verifications column.

    Arguments:
        data: The input dataframe.

    Returns:
        data: The output dataframe with formatted host_verifications.
    """
    # make a dictionary from the 'Host Verifications'
    verifications_counter = Counter()
    data["host_verifications"].astype("str").str.strip("[" "]").str.replace(
        "'", ""
    ).str.split(",").apply(verifications_counter.update)
    for item, _ in verifications_counter.most_common(10):
        col_name = "host_verifications" + item.replace(" ", "_")
        data[col_name] = (
            data["host_verifications"].astype("str").apply(lambda x: int(item in x))
        )
    data.drop(columns=["host_verifications"], axis=1, inplace=True)

    return data


def create_dummy_df(
    data: pd.DataFrame, categorical_cols: List[str], dummy_na: bool = True
):
    """
    This function is used to create a dummy dataframe from categorical columns
    having the following characteristics:

    1. contains all columns that were not specified as categorical
    2. dummy columns for each of the categorical columns in categorical_cols
    3. if dummy_na is True - it also contains dummy columns for the NaN values
    4. Use a prefix of the column name with an underscore (_) for separating

    Arguments:
        data: The input dataframe.
        categorical_cols: A list with the categorical columns.
        dummy_na: A boolean specifying whether we add dummy columns for NaNs.

    Returns:
        data: The output dataframe having the mentioned characteristics.

    """

    data = data.dropna(how="all", axis=1)

    for col in categorical_cols:
        try:
            # for each cat add dummy var, drop original column
            data = pd.concat(
                [
                    data.drop(col, axis=1),
                    pd.get_dummies(
                        data[col],
                        prefix=col,
                        prefix_sep="_",
                        drop_first=True,
                        dummy_na=dummy_na,
                    ),
                ],
                axis=1,
            )
        except:
            continue
    return data
