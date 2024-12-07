![Holberton School Logo](https://cdn.prod.website-files.com/6105315644a26f77912a1ada/63eea844ae4e3022154e2878_Holberton.png)

## Pandas Tutorial: Core Concepts and Key Functionalities  

Pandas is a powerful data analysis library in Python, widely used by data scientists and analysts to handle structured data. It provides robust tools for cleaning, transforming, and analyzing data, making it essential for any data-centric work.

This tutorial will introduce Pandas from the ground up, covering its core concepts, key methods, and functionalities.

---

### **1. Why Pandas?**

Modern data analysis requires handling various types of structured data, such as tables or time series. Pandas simplifies this process by providing:  
1. Easy data manipulation with intuitive commands.  
2. Handling of missing data.  
3. Integration with visualization libraries like matplotlib and seaborn.  
4. Compatibility with numerical libraries like NumPy.  

---

### **2. Pandas Data Structures**  

#### **2.1. Series**
A Pandas Series is a one-dimensional array-like structure that comes with labeled indices.

```python
import pandas as pd

# Creating a Series
data = [10, 20, 30, 40]
labels = ['a', 'b', 'c', 'd']
series = pd.Series(data, index=labels)

print(series)
```

*Output:*  
```
a    10
b    20
c    30
d    40
dtype: int64
```

You can access elements by label:
```python
print(series['b'])  # Output: 20
```

---

#### **2.2. DataFrame**  
The DataFrame is the centerpiece of Pandas, offering a two-dimensional, table-like structure. It has rows and columns with labels.

```python
# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)

print(df)
```

*Output:*  
```
      Name  Age  Salary
0    Alice   25   50000
1      Bob   30   60000
2  Charlie   35   70000
```

---

### **3. Exploring and Analyzing Data**

#### **3.1. Viewing Data**
Pandas provides convenient methods to inspect your data.  
```python
# Display the first 5 rows
print(df.head())

# Display the last 5 rows
print(df.tail())
```

#### **3.2. Summary Information**
Get an overview of your dataset's structure and statistics:  
```python
# Dataset info
print(df.info())

# Summary statistics
print(df.describe())
```

---

### **4. Selecting and Filtering Data**

#### **4.1. Selecting Columns**
Access columns using the column name:  
```python
# Select a single column
print(df['Age'])

# Select multiple columns
print(df[['Name', 'Salary']])
```

#### **4.2. Selecting Rows**
Rows can be accessed by index using `.iloc` or `.loc`:  
```python
# Select rows by position
print(df.iloc[1])  # Second row

# Select rows by label
df.index = ['A', 'B', 'C']  # Assign custom row labels
print(df.loc['B'])
```

#### **4.3. Filtering Rows**
Apply conditions to filter rows:  
```python
# Filter rows where Age > 30
print(df[df['Age'] > 30])

# Filter rows with multiple conditions
print(df[(df['Age'] > 25) & (df['Salary'] > 55000)])
```

---

### **5. Modifying Data**

#### **5.1. Adding Columns**
Add new columns by assigning values:
```python
# Add a new column
df['Bonus'] = df['Salary'] * 0.1
print(df)
```

#### **5.2. Modifying Existing Columns**
Update existing columns directly:
```python
# Update values in the Bonus column
df['Bonus'] = df['Bonus'] + 1000
print(df)
```

#### **5.3. Dropping Columns**
Remove unwanted columns:
```python
# Drop a column
df = df.drop(columns=['Bonus'])
print(df)
```

---

### **6. Handling Missing Data**

Real-world datasets often contain missing or null values. Pandas makes it easy to handle them.

#### **6.1. Identifying Missing Values**
```python
# Check for missing values
print(df.isnull())
print(df.isnull().sum())
```

#### **6.2. Filling Missing Values**
Replace missing values with a default value or calculation:
```python
# Fill missing values with 0
df.fillna(0, inplace=True)
```

#### **6.3. Dropping Missing Values**
Remove rows or columns with missing data:
```python
# Drop rows with missing values
df = df.dropna()
```

---

### **7. Grouping and Aggregating Data**

Pandas allows grouping of data for aggregation purposes, similar to SQL `GROUP BY`.

```python
# Example DataFrame
data = {
    'Department': ['HR', 'IT', 'HR', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David'],
    'Salary': [50000, 60000, 55000, 62000]
}
df = pd.DataFrame(data)

# Group by Department and calculate total salary
grouped = df.groupby('Department')['Salary'].sum()
print(grouped)
```

*Output:*  
```
Department
HR    105000
IT    122000
Name: Salary, dtype: int64
```

---

### **8. Merging and Joining DataFrames**

Pandas supports combining data from multiple DataFrames.

#### **8.1. Concatenation**
Concatenate DataFrames vertically or horizontally:  
```python
# Concatenate DataFrames vertically
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
result = pd.concat([df1, df2])
print(result)
```

#### **8.2. Merging**
Merge DataFrames based on a common key:
```python
# Example DataFrames
left = pd.DataFrame({'Key': ['A', 'B'], 'Value1': [1, 2]})
right = pd.DataFrame({'Key': ['A', 'C'], 'Value2': [3, 4]})
merged = pd.merge(left, right, on='Key', how='inner')
print(merged)
```

---

### **9. Reshaping Data**

Reshape the layout of your DataFrame for better analysis.

#### **9.1. Pivot Table**
Create a pivot table for multidimensional analysis:
```python
# Pivot table example
pivot = df.pivot_table(values='Salary', index='Department', aggfunc='mean')
print(pivot)
```

#### **9.2. Melting**
Convert wide-format data to long-format:
```python
# Melting a DataFrame
melted = pd.melt(df, id_vars=['Department'], value_vars=['Salary'])
print(melted)
```

---

### **10. Saving and Loading Data**

#### **10.1. Saving to a File**
```python
# Save to CSV
df.to_csv('data.csv', index=False)
```

#### **10.2. Loading from a File**
```python
# Load from CSV
df_loaded = pd.read_csv('data.csv')
```

---
