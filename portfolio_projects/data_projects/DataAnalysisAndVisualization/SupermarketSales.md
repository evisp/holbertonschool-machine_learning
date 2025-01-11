
![Holberton School Logo](https://cdn.prod.website-files.com/6105315644a26f77912a1ada/63eea844ae4e3022154e2878_Holberton.png)


## Supermarket Sales Dataset Analysis - Project Guide

---

#### **Project Overview:**
In this project, you will explore the **Supermarket Sales Dataset** available on Kaggle. The dataset provides detailed information about supermarket transactions, including the total purchase value, product categories, and customer demographics such as gender and customer type. The goal of this project is to perform **Exploratory Data Analysis (EDA)** using **Pandas** for data manipulation and **Matplotlib**/**Seaborn** for data visualization. By the end of the project, you will gain insights into purchasing behavior, product line preferences, payment methods, and demographic influences on sales.

#### **Project Requirements:**

1. **Get the Dataset:**
   - Download the Supermarket Sales Dataset from Kaggle using this link: [Supermarket Sales Dataset](https://www.kaggle.com/datasets/markmedhat/supermarket-sales?select=supermarket_sales+-+Sheet1.csv).
   - Load the dataset into a Pandas DataFrame using `pd.read_csv()`.

2. **Understand the Dataset:**
   - The dataset includes several columns such as:
     - **Invoice ID**, **Branch**, **City**, **Customer type**, **Gender**, **Product line**, **Unit price**, **Quantity**, **Tax 5%**, **Total**, **Date**, **Time**, **Payment**, **COGS (Cost of Goods Sold)**, **Gross margin percentage**, **Gross income**, **Rating**.
   - Each row represents a transaction, with columns detailing the transaction's total amount, the product(s) purchased, and the customer's demographic details.

3. **Tools Required:**
   - **Pandas**: For data manipulation and cleaning.
   - **Matplotlib** and **Seaborn**: For data visualization.
   - **NumPy** (optional for certain calculations).
   - If you are using `Jupyter notebooks` or `Google Colab` the above tools are already pre-installed 

#### **Project Objectives:**

- **Data Cleaning:**
   - Inspect the dataset for any missing or invalid values.
   - Handle missing data by filling, replacing, or dropping rows.
   - Convert columns with categorical data (e.g., **Gender**, **Payment**) into useful formats for analysis.

- **Exploratory Data Analysis (EDA):**
   - Analyze the distribution of key features like **Total**, **Quantity**, **Unit price**, **COGS**, and **Gender**.
   - Investigate relationships between features (e.g., **Product line** vs. **Total**).
   - Visualize the data using histograms, bar charts, and heatmaps to identify trends and outliers.

- **Key Insights:**
   - Explore how **gender**, **product line**, or **payment methods** influence total sales or purchasing patterns.
   - Investigate how ratings and customer type impact sales and product preferences.

---

#### **Key Questions to Guide Your Analysis:**

##### **Data Understanding:**
1. **What are the key columns in the supermarket dataset, and what data types do they represent?**
   - Begin by exploring the columns and data types using `df.info()` to understand the dataset structure.

2. **What is the distribution of total purchase values across the dataset?**
   - Use a histogram to explore the distribution of the `Total` column to understand typical transaction sizes.

3. **How is the quantity of products distributed in the dataset?**
   - Visualize the distribution of the `Quantity` column to explore how many items customers typically purchase.

4. **How do different product lines contribute to the total sales?**
   - Investigate the sales contribution from each product line using `groupby()` and visualize this with bar plots.

##### **Data Cleaning & Preprocessing:**
5. **How should missing values be handled in the dataset?**
   - Check for missing or null values using `df.isnull().sum()` and decide whether to fill, drop, or handle outliers appropriately.

6. **How can we handle categorical variables such as Gender, Payment, and Product line?**
   - Convert categorical columns into numerical formats using one-hot encoding (`pd.get_dummies()`) or label encoding.

##### **Exploratory Data Analysis (EDA):**
7. **What is the total sales breakdown by gender, and does it differ between male and female customers?**
   - Analyze total sales by gender using `groupby()` and visualize the differences using bar charts or pie charts.

8. **Which product lines are most frequently purchased, and what is the total sales contribution for each?**
   - Visualize the most popular product lines and their total sales contribution, using stacked bar charts or pie charts.

9. **How do payment methods influence the total sales, and which method is most frequently used?**
   - Explore the sales by different payment methods using `groupby()` and visualize the distribution with bar plots.

10. **What is the relationship between the quantity purchased and the total sales, and does this vary by product line?**
    - Investigate how quantity and total sales relate, potentially visualizing the correlation with scatter plots or line plots.

##### **Advanced Analysis:**
1. **How do ratings correlate with the total sales or customer spending behavior?**
    - Explore whether higher ratings impact sales and visualize the relationship using scatter plots or correlation matrices.

2. **Can we identify seasonal patterns in purchasing behavior (e.g., specific months, days, or times)?**
    - Analyze purchasing trends over time, using time-series analysis or month-wise/day-wise visualizations.

3. **Is there any significant difference in sales between customer types (Normal vs. Member)?**
    - Compare sales between regular customers and members using `groupby()` and visualize the results with bar plots or box plots.

4. **What impact do different cities have on the supermarket's overall sales?**
    - Explore the sales distribution across different cities and see if any city contributes more significantly to overall sales.

---

#### **Steps to Get Started:**
1. **Load the dataset:**
   - Use `pandas.read_csv()` to load the supermarket sales dataset into a Pandas DataFrame.

2. **Inspect the dataset:**
   - Use `df.head()` to preview the first few rows and `df.info()` to check the dataset structure, including data types and missing values.

3. **Clean the data:**
   - Handle missing values using `df.fillna()` or `df.dropna()`. Ensure categorical variables are encoded using `pd.get_dummies()`.

4. **Perform EDA:**
   - Explore individual features like `Total`, `Quantity`, `Product line`, `Gender`, `Payment`, and `Rating`.
   - Investigate the relationships between features using correlation matrices and visualizations like scatter plots, bar plots, and heatmaps.

5. **Answer the guiding questions:**
   - Use the key questions provided to ensure you address important aspects of the dataset and gain valuable insights.

6. **Summarize your findings:**
   - Conclude your analysis by summarizing key insights, such as which product lines generate the most revenue or how gender and payment methods influence purchasing behavior.

---