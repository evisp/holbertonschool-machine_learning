Here's an extended, detailed readme for your students, focusing on real-life use cases, the steps they'll take, and relevant database scripts. This readme is more comprehensive and aligns with the data analysis and segmentation project, removing JavaScript references:

---

# **Guided Project: Customer Analysis and Segmentation**

## **Project Overview**

In this project, you will work with both structured data from SQL (MySQL) and unstructured data from MongoDB to perform a customer analysis and segmentation. You’ll be using fake data from customer orders, transactions, and product information to create an analytical dashboard for business intelligence purposes.

### **Use Case Scenario**

Imagine you are working for a company that sells products online. The company wants to better understand their customer base to target different customer segments more effectively. By combining structured customer data with unstructured transaction data, you’ll be able to analyze customer spending habits, identify high-value customers, and generate actionable insights that could help the business optimize marketing efforts and product offerings.

## **Tools and Technologies**

1. **Visual Studio Code (VS Code)**: The primary IDE for writing Python scripts and interacting with the databases.
2. **Windows Subsystem for Linux (WSL)**: Allows you to run a Linux environment on your Windows machine.
3. **MySQL**: A relational database to store structured customer and transaction data.
4. **MongoDB**: A NoSQL database to store unstructured transaction logs and customer interactions.
5. **Python**: For data processing, cleaning, and merging from both databases.
6. **Pandas**: Python library used for data analysis and manipulation.

---

## **Project Goals**

The primary goals of this project are:
1. **Data Integration**: Integrate structured data (customer details, transactions) from MySQL with unstructured transaction logs from MongoDB.
2. **Data Cleaning**: Use Python and Pandas to clean and transform the data into a format suitable for analysis.
3. **Customer Segmentation**: Segment customers based on their spending and transaction frequency, identifying high-value customers.
4. **Real-world Use Cases**: Apply real business scenarios, such as identifying key customer segments and predicting future buying behavior.
5. **Data Visualization**: Generate simple insights using Python, showcasing the process of merging and analyzing data.

---

## **Step-by-Step Guidance**

### **Setup MySQL Database (Structured Data)**

In this step, you’ll work with the structured data, which includes customer details, product information, and transactions. Use the following SQL scripts to create the necessary database and populate it with sample data.

1. **Create Database**: 
   - The script `create_customer_analysis_db.sql` will create the necessary schema for the analysis.
     ```bash
     mysql -u root -p < create_customer_analysis_db.sql
     ```

2. **Populate the Database**:
   - Run the following scripts to populate the `customers`, `products`, and `transactions` tables:
     - `populate_customers.sql`: Inserts sample customer data into the `customers` table.
     - `populate_products.sql`: Inserts sample product data into the `products` table.
     - `populate_transactions.sql`: Inserts sample transaction data into the `transactions` table.
     
     Run each script using MySQL:
     ```bash
     mysql -u root -p < populate_customers.sql
     mysql -u root -p < populate_products.sql
     mysql -u root -p < populate_transactions.sql
     ```

3. **Verify Data**:
   - To confirm the data was inserted correctly, you can run the following query to view the customer data:
     ```sql
     SELECT * FROM customers;
     ```

---

### **Setup MongoDB Database (Unstructured Data)**

In this step, you’ll work with MongoDB to store customer order logs in a more flexible format (unstructured data).

1. **Populate MongoDB**:
   - Use the `populate_mongo.sh` script to populate MongoDB with transaction data.
   - First, ensure MongoDB is running:
     ```bash
     sudo service mongodb start
     ```
   - Then, run the script:
     ```bash
     bash populate_mongo.sh
     ```

2. **Verify Data**:
   - You can check the data inside MongoDB using the `mongo` shell:
     ```bash
     mongo
     use customer_orders
     db.orders.find().pretty();
     ```

---

### **Data Cleaning and Transformation with Python**

In this step, you will extract, transform, and load data from both MySQL and MongoDB databases into a unified format, which will be saved as a CSV file for further analysis.

#### **Goals:**
- Connect to both MySQL and MongoDB databases.
- Extract data from customers, products, transactions, customer reviews, and product social comments.
- Clean and merge the data from both databases into a single DataFrame.
- Save the combined data into a CSV file for future analysis.

#### **Tools Required:**
- Python (preferably 3.x)
- Libraries: `mysql-connector-python`, `pymongo`, `pandas`
- MySQL and MongoDB running locally or remotely

#### **Key Steps:**

1. **MySQL Database Connection:**
   First, establish a connection to your MySQL database. You will use this connection to query data from three tables: customers, products, and transactions.
   
   ```python
   mysql_connection = mysql.connector.connect(
       host="localhost",
       user="root",
       password="root",
       database="customer_analysis"
   )
   ```

2. **MongoDB Database Connection:**
   Similarly, connect to your MongoDB database to extract customer reviews and product social comments. The collections you need are `customer_reviews` and `product_social_comments`.
   
   ```python
   client = pymongo.MongoClient("mongodb://localhost:27017/")
   db = client["customer_analysis_mongo"]
   customer_reviews_collection = db["customer_reviews"]
   product_social_comments_collection = db["product_social_comments"]
   ```

3. **Extract Data from MySQL:**
   Query the necessary data from the MySQL database tables using SQL queries. You will extract customer information, product details, and transaction data.
   
   ```python
   customers_df = pd.read_sql("SELECT customer_id, name, email, date_of_birth, registration_date FROM customers;", mysql_connection)
   products_df = pd.read_sql("SELECT product_id, name, category, price FROM products;", mysql_connection)
   transactions_df = pd.read_sql("SELECT transaction_id, customer_id, product_id, quantity, transaction_date FROM transactions;", mysql_connection)
   ```

4. **Extract Data from MongoDB:**
   Retrieve data from MongoDB collections (`customer_reviews` and `product_social_comments`). This data will be loaded into pandas DataFrames for easier merging and cleaning.
   
   ```python
   mongo_reviews_data = pd.DataFrame(list(customer_reviews_collection.find()))
   mongo_social_comments_data = pd.DataFrame(list(product_social_comments_collection.find()))
   ```

5. **Data Transformation:**
   - Rename columns in the DataFrames to avoid conflicts and ensure consistent naming.
   - Merge the data from different sources into a single unified DataFrame.

   ```python
   products_df.rename(columns={'name': 'product_name'}, inplace=True)
   mongo_reviews_data.rename(columns={'review_text': 'review_text', 'rating': 'review_rating'}, inplace=True)
   mongo_social_comments_data.rename(columns={'comment_text': 'comment_text', 'sentiment': 'comment_sentiment'}, inplace=True)
   ```

6. **Data Merging:**
   - Merge the customer and transaction data.
   - Merge the product data with transaction details.
   - Merge customer reviews and social comments with product data.

   ```python
   customer_transactions_df = pd.merge(customers_df, transactions_df, on="customer_id", how="inner")
   customer_transactions_product_df = pd.merge(customer_transactions_df, products_df, on="product_id", how="inner")
   reviews_with_product_df = pd.merge(mongo_reviews_data, products_df, on="product_id", how="inner")
   social_comments_with_product_df = pd.merge(mongo_social_comments_data, products_df, on="product_id", how="inner")
   ```

7. **Final DataFrame:**
   Combine all the data into one final DataFrame, merging customer transactions, product information, reviews, and social comments.
   
   ```python
   final_df = pd.merge(customer_transactions_product_df, reviews_with_product_df, on=["customer_id", "product_id"], how="left")
   final_df = pd.merge(final_df, social_comments_with_product_df, on=["customer_id", "product_id"], how="left", suffixes=('_review', '_comment'))
   ```

8. **Save the Final Data:**
   The final DataFrame, which now contains merged and cleaned data from both MySQL and MongoDB, is saved to a CSV file.

   ```python
   final_df.to_csv('customer_analysis_combined_data.csv', index=False)
   print("Data saved to 'customer_analysis_combined_data.csv'")
   ```

#### **Next Steps:**
Once the data is saved to a CSV file, you can move on to analyzing and visualizing the data. The next steps may include customer segmentation, analyzing purchase patterns, or conducting sentiment analysis on product reviews and social comments.


## **Conclusion**

You’ve successfully integrated structured data from MySQL and unstructured data from MongoDB, cleaned and merged them, and performed basic customer segmentation. This process mirrors real-world scenarios where businesses combine different data sources for in-depth analysis. You can now use these insights to make data-driven decisions for marketing, sales strategies, and customer retention.

### **Key Takeaways**
- Integration of structured and unstructured data.
- Basic data cleaning and merging with Python.
- Customer segmentation based on spending and transaction frequency.

---

## **References**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [MySQL Documentation](https://dev.mysql.com/doc/)

