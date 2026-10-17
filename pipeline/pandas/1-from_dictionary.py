#!/usr/bin/env python3
"""
Build a pandas DataFrame from a Python dictionary.

The module defines `df`, a 4x2 DataFrame with the columns "First" (numbers)
and "Second" (words) and the row labels A, B, C and D.
"""
import pandas as pd


# Each key becomes a column and each list holds that column's values
d = {'First': [0, 0.5, 1, 1.5], 'Second': ['one', 'two', 'three', 'four']}
# Label the rows A to D instead of the default 0 to 3
df = pd.DataFrame(d, index=list('ABCD'))
