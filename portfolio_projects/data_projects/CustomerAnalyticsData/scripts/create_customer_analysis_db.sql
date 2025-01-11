-- Create the customer_analysis database
CREATE DATABASE IF NOT EXISTS customer_analysis;

-- Use the customer_analysis database
USE customer_analysis;

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    date_of_birth DATE,
    registration_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(255),
    price DECIMAL(10, 2)
);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    quantity INT,
    transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
