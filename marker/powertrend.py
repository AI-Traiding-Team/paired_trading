import pandas as pd
import numpy as np

__version__ = 0.0002


def calc_power_trend(input_df: pd.DataFrame,
                     W: float = 0.15
                     ) -> pd.DataFrame:
    """
    Args:
        input_df (pd.DataFrame):    input DataFrame with index and OHCLV data
        W (float):                  weight percentage for calc trend default 0.15
    Returns:
       trend (pd.DataFrame):        output DataFrame
    """

    """
    Setup. Getting first data and initialize variables
    """
    x_bar_0 = [input_df.index[0],  # [0] - datetime
               input_df["Open"][0],  # [1] - open
               input_df["High"][0],  # [2] - high
               input_df["Low"][0],  # [3] - low
               input_df["Close"][0],  # [4] - CLOSE
               ]
    FP_first_price = x_bar_0[4]
    xH_highest_price = x_bar_0[2]
    HT_highest_price_timemark = 0
    xL_lowest_price = x_bar_0[3]
    LT_lowest_price_timemark = 0
    Cid = 0
    FPN_first_price_idx = 0
    Cid_array = np.zeros(input_df.shape[0])
    """
    Setup. Getting first data and initialize variables
    """

    for idx in range(input_df.shape[0] - 1):
        x_bar = [input_df.index[idx],
                 input_df["Open"][idx],
                 input_df["High"][idx],
                 input_df["Low"][idx],
                 input_df["Close"][idx],
                 ]
        # print(x_bar)
        # print(x_bar[4])
        if x_bar[2] > (FP_first_price + x_bar_0[4] * W):
            xH_highest_price = x_bar[2]
            HT_highest_price_timemark = idx
            FPN_first_price_idx = idx
            Cid = 1
            Cid_array[idx] = 1
            Cid_array[0] = 1
            break
        if x_bar[3] < (FP_first_price - x_bar_0[4] * W):
            xL_lowest_price = x_bar[3]
            LT_lowest_price_timemark = idx
            FPN_first_price_idx = idx
            Cid = -1
            Cid_array[idx] = -1
            Cid_array[0] = -1
            break

    for ix in range(FPN_first_price_idx + 1, input_df.shape[0] - 2):
        x_bar = [input_df.index[ix],
                 input_df["Open"][ix],
                 input_df["High"][ix],
                 input_df["Low"][ix],
                 input_df["Close"][ix],
                 ]
        if Cid > 0:
            if x_bar[2] > xH_highest_price:
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
            if x_bar[2] < (
                    xH_highest_price - xH_highest_price * W) and LT_lowest_price_timemark <= HT_highest_price_timemark:
                for j in range(1, input_df.shape[0] - 1):
                    if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                        Cid_array[j] = 1
                xL_lowest_price = x_bar[2]
                LT_lowest_price_timemark = ix
                Cid = -1

        if Cid < 0:
            if x_bar[3] < xL_lowest_price:
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = ix
            if x_bar[3] > (
                    xL_lowest_price + xL_lowest_price * W) and HT_highest_price_timemark <= LT_lowest_price_timemark:
                for j in range(1, input_df.shape[0] - 1):
                    if HT_highest_price_timemark < j <= LT_lowest_price_timemark:
                        Cid_array[j] = -1
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
                Cid = 1

    # TODO: rewrite this block in intelligent way !!! Now is working but code is ugly
    """ Checking last bar in input_df """
    ix = input_df.shape[0] - 1
    x_bar = [input_df.index[ix],
             input_df["Open"][ix],
             input_df["High"][ix],
             input_df["Low"][ix],
             input_df["Close"][ix],
             ]
    if Cid > 0:
        if x_bar[2] > xH_highest_price:
            xH_highest_price = x_bar[2]
            HT_highest_price_timemark = ix
        if x_bar[2] <= xH_highest_price:
            for j in range(1, input_df.shape[0]):
                if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                    Cid_array[j] = 1
            xL_lowest_price = x_bar[3]
            LT_lowest_price_timemark = ix
            Cid = -1
    if Cid < 0:
        if x_bar[3] < xL_lowest_price:
            xL_lowest_price = x_bar[3]
            LT_lowest_price_timemark = ix
            # print(True)
        if x_bar[3] >= xL_lowest_price:
            for j in range(1, input_df.shape[0]):
                if HT_highest_price_timemark < j <= LT_lowest_price_timemark:
                    Cid_array[j] = -1
            xH_highest_price = x_bar[2]
            HT_highest_price_timemark = ix
            Cid = 1
    if Cid > 0:
        if x_bar[2] > xH_highest_price:
            xH_highest_price = x_bar[2]
            HT_highest_price_timemark = ix
        if x_bar[2] <= xH_highest_price:
            for j in range(1, input_df.shape[0]):
                if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                    Cid_array[j] = 1
            # xL_lowest_price = x_bar[3]
            # LT_lowest_price_timemark = ix
            # Cid = -1
    trend = pd.DataFrame(data=Cid_array,
                         index=input_df.index,
                         columns=["trend"])
    return trend