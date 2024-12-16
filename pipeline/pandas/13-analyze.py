#!/usr/bin/env python3
"""
Task 14
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the 'Weighted_Price' column
df = df.drop(columns=['Weighted_Price'])

# Rename 'Timestamp' to 'Date' and convert to datetime
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on 'Date'
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data to include only entries from 2017 onwards
df = df.loc[df.index >= '2017-01-01']

# Resample data at daily intervals and aggregate
daily_data = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

print(daily_data)

# Plot the daily data
daily_data.plot()
plt.show()
