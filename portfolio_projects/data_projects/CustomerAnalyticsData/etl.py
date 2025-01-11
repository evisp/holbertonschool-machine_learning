import mysql.connector
import pymongo
import pandas as pd

# MySQL Connection
mysql_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="customer_analysis"
)

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["customer_analysis_mongo"]
customer_reviews_collection = db["customer_reviews"]
product_social_comments_collection = db["product_social_comments"]

# Query MySQL database for customers, products, and transactions
mysql_customer_query = """
SELECT customer_id, name, email, date_of_birth, registration_date 
FROM customers;
"""
mysql_products_query = """
SELECT product_id, name, category, price 
FROM products;
"""
mysql_transactions_query = """
SELECT transaction_id, customer_id, product_id, quantity, transaction_date 
FROM transactions;
"""

# Load data from MySQL into pandas DataFrames
customers_df = pd.read_sql(mysql_customer_query, mysql_connection)
products_df = pd.read_sql(mysql_products_query, mysql_connection)
transactions_df = pd.read_sql(mysql_transactions_query, mysql_connection)

# Query MongoDB collections for reviews and social comments
mongo_reviews_data = pd.DataFrame(list(customer_reviews_collection.find()))
mongo_social_comments_data = pd.DataFrame(list(product_social_comments_collection.find()))

# Rename columns to avoid conflicts
products_df.rename(columns={'name': 'product_name'}, inplace=True)
mongo_reviews_data.rename(columns={'review_text': 'review_text', 'rating': 'review_rating'}, inplace=True)
mongo_social_comments_data.rename(columns={'comment_text': 'comment_text', 'sentiment': 'comment_sentiment'}, inplace=True)

# Merge SQL data into a single DataFrame
# Merging customers and transactions
customer_transactions_df = pd.merge(customers_df, transactions_df, on="customer_id", how="inner")

# Merging the customer_transactions_df with products
customer_transactions_product_df = pd.merge(customer_transactions_df, products_df, on="product_id", how="inner")

# Merge MongoDB reviews and social comments with product data
reviews_with_product_df = pd.merge(mongo_reviews_data, products_df, on="product_id", how="inner")
social_comments_with_product_df = pd.merge(mongo_social_comments_data, products_df, on="product_id", how="inner")

# Combine all data into one DataFrame
final_df = pd.merge(customer_transactions_product_df, reviews_with_product_df, on=["customer_id", "product_id"], how="left")
final_df = pd.merge(final_df, social_comments_with_product_df, on=["customer_id", "product_id"], how="left", suffixes=('_review', '_comment'))

# Save the final DataFrame to a CSV file
final_df.to_csv('customer_analysis_combined_data.csv', index=False)

print("Data saved to 'customer_analysis_combined_data.csv'")


# Close MySQL connection
mysql_connection.close()
