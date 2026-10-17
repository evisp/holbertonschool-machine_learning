#!/usr/bin/env python3
"""
Fill the missing values of a price DataFrame.
"""


def fill(df):
    """
    Clean a price DataFrame by filling its missing values.

    Args:
        df: a DataFrame with the columns "Weighted_Price", "Close", "High",
            "Low", "Open", "Volume_(BTC)" and "Volume_(Currency)".

    Returns:
        The DataFrame without "Weighted_Price" and with no missing values:
        - "Close" takes the previous row's value when missing
        - "High", "Low" and "Open" take the same row's "Close" when missing
        - the two volume columns become 0 when missing, since no trade
          happened
    """
    # This column is not needed
    df = df.drop(columns='Weighted_Price')
    # Carry the last known close forward
    df['Close'] = df['Close'].ffill(axis=0)

    # With no trades in a minute, every price equals the close
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # No trades means no volume
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
