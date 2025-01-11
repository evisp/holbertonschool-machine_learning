#!/bin/bash

# Define the MongoDB database
DB_NAME="customer_analysis_mongo"

# Connect to MongoDB and populate collections
mongo $DB_NAME <<EOF
// Create customer_reviews collection and insert data
db.customer_reviews.insertMany([
    { review_id: 1, customer_id: 1, product_id: 1, review_text: "Excellent product, would definitely recommend!", rating: 5 },
    { review_id: 2, customer_id: 2, product_id: 2, review_text: "Not what I expected, quality is lacking.", rating: 2 },
    { review_id: 3, customer_id: 3, product_id: 3, review_text: "Perfect for everyday use. Will buy again.", rating: 4 },
    { review_id: 4, customer_id: 4, product_id: 4, review_text: "It's decent but not worth the price.", rating: 3 },
    { review_id: 5, customer_id: 5, product_id: 5, review_text: "Amazing! Loved it, exactly what I needed.", rating: 5 },
    { review_id: 6, customer_id: 6, product_id: 6, review_text: "Very poor. Doesn't work as advertised.", rating: 1 },
    { review_id: 7, customer_id: 7, product_id: 7, review_text: "Good, but can be improved with better materials.", rating: 3 },
    { review_id: 8, customer_id: 8, product_id: 8, review_text: "Fantastic quality, I am very happy.", rating: 5 },
    { review_id: 9, customer_id: 9, product_id: 9, review_text: "It's okay, meets expectations.", rating: 4 },
    { review_id: 10, customer_id: 10, product_id: 10, review_text: "Not satisfied with the durability.", rating: 2 },
    { review_id: 11, customer_id: 11, product_id: 11, review_text: "Exceeded my expectations in every way.", rating: 5 },
    { review_id: 12, customer_id: 12, product_id: 12, review_text: "Good but too expensive for what it offers.", rating: 3 },
    { review_id: 13, customer_id: 13, product_id: 13, review_text: "Average product, not terrible but not great either.", rating: 3 },
    { review_id: 14, customer_id: 14, product_id: 14, review_text: "I absolutely love this product!", rating: 5 },
    { review_id: 15, customer_id: 15, product_id: 15, review_text: "Very bad. Didn’t live up to the hype.", rating: 1 },
    { review_id: 16, customer_id: 16, product_id: 16, review_text: "Great item, exactly what I needed.", rating: 5 },
    { review_id: 17, customer_id: 17, product_id: 17, review_text: "It works fine but not as fast as I expected.", rating: 3 },
    { review_id: 18, customer_id: 18, product_id: 18, review_text: "Great product, but it could use some improvements.", rating: 4 },
    { review_id: 19, customer_id: 19, product_id: 19, review_text: "Works well, easy to use. Worth the money.", rating: 4 },
    { review_id: 20, customer_id: 20, product_id: 20, review_text: "It broke within a week. Very disappointing.", rating: 1 },
    { review_id: 21, customer_id: 21, product_id: 1, review_text: "Perfect for everyday tasks, solid product!", rating: 5 },
    { review_id: 22, customer_id: 22, product_id: 2, review_text: "Quality could be better, but it's decent for the price.", rating: 3 },
    { review_id: 23, customer_id: 23, product_id: 3, review_text: "Nice product, but I think it can be improved.", rating: 4 },
    { review_id: 24, customer_id: 24, product_id: 4, review_text: "Good product, not great though.", rating: 3 },
    { review_id: 25, customer_id: 25, product_id: 5, review_text: "Totally worth the price! Very satisfied.", rating: 5 },
    { review_id: 26, customer_id: 26, product_id: 6, review_text: "Didn’t meet my expectations, very poor quality.", rating: 1 },
    { review_id: 27, customer_id: 27, product_id: 7, review_text: "Great quality, but a little pricey.", rating: 4 },
    { review_id: 28, customer_id: 28, product_id: 8, review_text: "Fantastic, I love it!", rating: 5 },
    { review_id: 29, customer_id: 29, product_id: 9, review_text: "It’s an average product, not special.", rating: 3 },
    { review_id: 30, customer_id: 30, product_id: 10, review_text: "Not worth the money, poor durability.", rating: 2 },
    { review_id: 31, customer_id: 31, product_id: 11, review_text: "This product is top-notch, I’m impressed!", rating: 5 },
    { review_id: 32, customer_id: 32, product_id: 12, review_text: "Good value for money.", rating: 4 },
    { review_id: 33, customer_id: 33, product_id: 13, review_text: "Meh, it does the job but could be improved.", rating: 3 },
    { review_id: 34, customer_id: 34, product_id: 14, review_text: "Very good, I would recommend it.", rating: 5 },
    { review_id: 35, customer_id: 35, product_id: 15, review_text: "Worst purchase I’ve made in a long time.", rating: 1 },
    { review_id: 36, customer_id: 36, product_id: 16, review_text: "Worth every penny!", rating: 5 },
    { review_id: 37, customer_id: 37, product_id: 17, review_text: "Product didn’t meet my expectations, would not buy again.", rating: 2 },
    { review_id: 38, customer_id: 38, product_id: 18, review_text: "It’s an okay product, but I expected more.", rating: 3 },
    { review_id: 39, customer_id: 39, product_id: 19, review_text: "Superb quality, highly recommend.", rating: 5 },
    { review_id: 40, customer_id: 40, product_id: 20, review_text: "The product arrived broken, totally unacceptable.", rating: 1 },
    { review_id: 41, customer_id: 41, product_id: 1, review_text: "Satisfactory, does the job but could be improved.", rating: 3 },
    { review_id: 42, customer_id: 42, product_id: 2, review_text: "Better than expected, would buy again.", rating: 4 },
    { review_id: 43, customer_id: 43, product_id: 3, review_text: "Nice product but needs more features.", rating: 3 },
    { review_id: 44, customer_id: 44, product_id: 4, review_text: "Loved it, but it could be better in terms of size.", rating: 4 },
    { review_id: 45, customer_id: 45, product_id: 5, review_text: "Would definitely recommend to my friends.", rating: 5 },
    { review_id: 46, customer_id: 46, product_id: 6, review_text: "Worst product I have ever bought, do not recommend.", rating: 1 },
    { review_id: 47, customer_id: 47, product_id: 7, review_text: "Excellent quality and functionality, worth the price.", rating: 5 },
    { review_id: 48, customer_id: 48, product_id: 8, review_text: "Very good, meets expectations.", rating: 4 },
    { review_id: 49, customer_id: 49, product_id: 9, review_text: "Not bad, works well.", rating: 3 },
    { review_id: 50, customer_id: 50, product_id: 10, review_text: "Terrible product, broke after two uses.", rating: 1 },
    { review_id: 51, customer_id: 51, product_id: 11, review_text: "Exceeded my expectations, amazing quality!", rating: 5 },
    { review_id: 52, customer_id: 52, product_id: 12, review_text: "Good, but expected more features.", rating: 3 },
    { review_id: 53, customer_id: 53, product_id: 13, review_text: "Not worth the price, lacks durability.", rating: 2 },
    { review_id: 54, customer_id: 54, product_id: 14, review_text: "Fantastic purchase, very happy with it.", rating: 5 },
    { review_id: 55, customer_id: 55, product_id: 15, review_text: "Totally unreliable. I regret this purchase.", rating: 1 },
    { review_id: 56, customer_id: 56, product_id: 16, review_text: "Great product, does exactly what I need.", rating: 5 },
    { review_id: 57, customer_id: 57, product_id: 17, review_text: "It works, but it’s slower than expected.", rating: 3 },
    { review_id: 58, customer_id: 58, product_id: 18, review_text: "Highly recommend! Will buy again.", rating: 5 },
    { review_id: 59, customer_id: 59, product_id: 19, review_text: "Very good, fits all my needs.", rating: 4 },
    { review_id: 60, customer_id: 60, product_id: 20, review_text: "Didn’t work at all, very disappointing.", rating: 1 },
    { review_id: 61, customer_id: 61, product_id: 1, review_text: "Great quality, but a bit expensive.", rating: 4 },
    { review_id: 62, customer_id: 62, product_id: 2, review_text: "Not bad, but could be improved.", rating: 3 },
    { review_id: 63, customer_id: 63, product_id: 3, review_text: "Absolutely love it, works perfectly!", rating: 5 },
    { review_id: 64, customer_id: 64, product_id: 4, review_text: "Quality isn’t as good as expected.", rating: 2 },
    { review_id: 65, customer_id: 65, product_id: 5, review_text: "Exceeded expectations. Highly recommend.", rating: 5 },
    { review_id: 66, customer_id: 66, product_id: 6, review_text: "Quite average. I expected more.", rating: 3 },
    { review_id: 67, customer_id: 67, product_id: 7, review_text: "This is fantastic. Will definitely buy again!", rating: 5 },
    { review_id: 68, customer_id: 68, product_id: 8, review_text: "Doesn’t fit well, returned it.", rating: 2 },
    { review_id: 69, customer_id: 69, product_id: 9, review_text: "Great value for money.", rating: 4 },
    { review_id: 70, customer_id: 70, product_id: 10, review_text: "Not very durable. I wouldn’t recommend it.", rating: 1 },
    { review_id: 71, customer_id: 71, product_id: 11, review_text: "Fits perfectly and looks great!", rating: 5 },
    { review_id: 72, customer_id: 72, product_id: 12, review_text: "Very poor quality, won’t buy again.", rating: 1 },
    { review_id: 73, customer_id: 73, product_id: 13, review_text: "It’s alright, but didn’t meet all expectations.", rating: 3 },
    { review_id: 74, customer_id: 74, product_id: 14, review_text: "I’m so impressed. Best purchase ever!", rating: 5 },
    { review_id: 75, customer_id: 75, product_id: 15, review_text: "Decent, but there are better alternatives.", rating: 3 },
    { review_id: 76, customer_id: 76, product_id: 16, review_text: "Not as described. Very disappointed.", rating: 2 },
    { review_id: 77, customer_id: 77, product_id: 17, review_text: "Perfect for what I needed.", rating: 4 },
    { review_id: 78, customer_id: 78, product_id: 18, review_text: "Terrible product. It broke immediately.", rating: 1 },
    { review_id: 79, customer_id: 79, product_id: 19, review_text: "It works, but very slow.", rating: 3 },
    { review_id: 80, customer_id: 80, product_id: 20, review_text: "Very good overall. Great for the price.", rating: 4 },
    { review_id: 81, customer_id: 81, product_id: 1, review_text: "Not as good as expected. Wouldn’t recommend.", rating: 2 },
    { review_id: 82, customer_id: 82, product_id: 2, review_text: "Good product, but quality could be improved.", rating: 3 },
    { review_id: 83, customer_id: 83, product_id: 3, review_text: "Very satisfied with this product.", rating: 5 },
    { review_id: 84, customer_id: 84, product_id: 4, review_text: "Average. Works, but lacks certain features.", rating: 3 },
    { review_id: 85, customer_id: 85, product_id: 5, review_text: "Fantastic quality and performance!", rating: 5 },
    { review_id: 86, customer_id: 86, product_id: 6, review_text: "Wouldn’t buy again. It broke after a month.", rating: 2 },
    { review_id: 87, customer_id: 87, product_id: 7, review_text: "Love the design, but it’s a bit noisy.", rating: 4 },
    { review_id: 88, customer_id: 88, product_id: 8, review_text: "Not worth the price. Very poor quality.", rating: 1 },
    { review_id: 89, customer_id: 89, product_id: 9, review_text: "Great product, exceeded my expectations.", rating: 5 },
    { review_id: 90, customer_id: 90, product_id: 10, review_text: "It’s decent but not the best in its class.", rating: 3 },
    { review_id: 91, customer_id: 91, product_id: 11, review_text: "Awesome! I’m very happy with this purchase.", rating: 5 },
    { review_id: 92, customer_id: 92, product_id: 12, review_text: "Waste of money. Very disappointing.", rating: 1 },
    { review_id: 93, customer_id: 93, product_id: 13, review_text: "Okay product, but feels cheap.", rating: 3 },
    { review_id: 94, customer_id: 94, product_id: 14, review_text: "I love it. It works perfectly!", rating: 5 },
    { review_id: 95, customer_id: 95, product_id: 15, review_text: "Good product, but not as described.", rating: 2 }
]);

// Create product_social_comments collection and insert data
db.product_social_comments.insertMany([
    { comment_id: 1, product_id: 1, customer_id: 1, platform: "Facebook", comment_text: "This product is amazing!", sentiment: "positive" },
    { comment_id: 2, product_id: 2, customer_id: 2, platform: "Twitter", comment_text: "Not happy with this at all. Too expensive.", sentiment: "negative" },
    { comment_id: 3, product_id: 3, customer_id: 3, platform: "Instagram", comment_text: "Love it, definitely a must-have!", sentiment: "positive" },
    { comment_id: 4, product_id: 4, customer_id: 4, platform: "LinkedIn", comment_text: "Not worth the price, but decent.", sentiment: "neutral" },
    { comment_id: 5, product_id: 5, customer_id: 5, platform: "Snapchat", comment_text: "Absolutely fantastic!", sentiment: "positive" },
    { comment_id: 6, product_id: 6, customer_id: 6, platform: "Reddit", comment_text: "Terrible product, doesn't work.", sentiment: "negative" },
    { comment_id: 7, product_id: 7, customer_id: 7, platform: "Facebook", comment_text: "It’s a good product, but overpriced.", sentiment: "neutral" },
    { comment_id: 8, product_id: 8, customer_id: 8, platform: "Twitter", comment_text: "So impressed by the quality!", sentiment: "positive" },
    { comment_id: 9, product_id: 9, customer_id: 9, platform: "Instagram", comment_text: "It's okay, but could be improved.", sentiment: "neutral" },
    { comment_id: 10, product_id: 10, customer_id: 10, platform: "LinkedIn", comment_text: "Broke after one week of use. Do not buy.", sentiment: "negative" },
    { comment_id: 11, product_id: 11, customer_id: 11, platform: "Facebook", comment_text: "Excellent value for the price!", sentiment: "positive" },
    { comment_id: 12, product_id: 12, customer_id: 12, platform: "Twitter", comment_text: "Not what I expected, very disappointing.", sentiment: "negative" },
    { comment_id: 13, product_id: 13, customer_id: 13, platform: "Instagram", comment_text: "Perfect for my needs, highly recommend.", sentiment: "positive" },
    { comment_id: 14, product_id: 14, customer_id: 14, platform: "LinkedIn", comment_text: "Decent, but there are better options.", sentiment: "neutral" },
    { comment_id: 15, product_id: 15, customer_id: 15, platform: "Snapchat", comment_text: "Great purchase, works as advertised!", sentiment: "positive" },
    { comment_id: 16, product_id: 16, customer_id: 16, platform: "Reddit", comment_text: "Waste of money. Doesn’t function as described.", sentiment: "negative" },
    { comment_id: 17, product_id: 17, customer_id: 17, platform: "Facebook", comment_text: "Good, but lacks some essential features.", sentiment: "neutral" },
    { comment_id: 18, product_id: 18, customer_id: 18, platform: "Twitter", comment_text: "Very happy with this purchase, works great!", sentiment: "positive" },
    { comment_id: 19, product_id: 19, customer_id: 19, platform: "Instagram", comment_text: "It’s okay, but not great. Could be better.", sentiment: "neutral" },
    { comment_id: 20, product_id: 20, customer_id: 20, platform: "LinkedIn", comment_text: "Extremely poor quality, returned it immediately.", sentiment: "negative" },
    { comment_id: 21, product_id: 1, customer_id: 21, platform: "Facebook", comment_text: "Love this product, worth every penny!", sentiment: "positive" },
    { comment_id: 22, product_id: 2, customer_id: 22, platform: "Twitter", comment_text: "Doesn’t work as expected. Very disappointed.", sentiment: "negative" },
    { comment_id: 23, product_id: 3, customer_id: 23, platform: "Instagram", comment_text: "Perfect fit, highly recommend it!", sentiment: "positive" },
    { comment_id: 24, product_id: 4, customer_id: 24, platform: "LinkedIn", comment_text: "Okay, but I expected better quality for the price.", sentiment: "neutral" },
    { comment_id: 25, product_id: 5, customer_id: 25, platform: "Snapchat", comment_text: "Absolutely love this! Worth every cent.", sentiment: "positive" },
    { comment_id: 26, product_id: 6, customer_id: 26, platform: "Reddit", comment_text: "Complete waste of money. Doesn’t work at all.", sentiment: "negative" },
    { comment_id: 27, product_id: 7, customer_id: 27, platform: "Facebook", comment_text: "Good product, but a bit too expensive.", sentiment: "neutral" },
    { comment_id: 28, product_id: 8, customer_id: 28, platform: "Twitter", comment_text: "So impressed, exceeded my expectations!", sentiment: "positive" },
    { comment_id: 29, product_id: 9, customer_id: 29, platform: "Instagram", comment_text: "Not bad, but it could use some improvements.", sentiment: "neutral" },
    { comment_id: 30, product_id: 10, customer_id: 30, platform: "LinkedIn", comment_text: "Terrible quality. Do not buy this product.", sentiment: "negative" },
    { comment_id: 31, product_id: 11, customer_id: 31, platform: "Facebook", comment_text: "Highly recommend. Works like a charm!", sentiment: "positive" },
    { comment_id: 32, product_id: 12, customer_id: 32, platform: "Twitter", comment_text: "Disappointed with the purchase. Not worth the price.", sentiment: "negative" },
    { comment_id: 33, product_id: 13, customer_id: 33, platform: "Instagram", comment_text: "Really happy with it! Great performance.", sentiment: "positive" },
    { comment_id: 34, product_id: 14, customer_id: 34, platform: "LinkedIn", comment_text: "Just okay, didn’t meet all my expectations.", sentiment: "neutral" },
    { comment_id: 35, product_id: 15, customer_id: 35, platform: "Snapchat", comment_text: "Love it! Amazing quality for the price.", sentiment: "positive" },
    { comment_id: 36, product_id: 16, customer_id: 36, platform: "Reddit", comment_text: "Horrible experience, broke in a week.", sentiment: "negative" },
    { comment_id: 37, product_id: 17, customer_id: 37, platform: "Facebook", comment_text: "Good but overpriced. I expected more for the price.", sentiment: "neutral" },
    { comment_id: 38, product_id: 18, customer_id: 38, platform: "Twitter", comment_text: "Great product, works as expected.", sentiment: "positive" },
    { comment_id: 39, product_id: 19, customer_id: 39, platform: "Instagram", comment_text: "It’s decent, but I’m not thrilled.", sentiment: "neutral" },
    { comment_id: 40, product_id: 20, customer_id: 40, platform: "LinkedIn", comment_text: "Terrible, don’t waste your money on this.", sentiment: "negative" },
    { comment_id: 41, product_id: 1, customer_id: 41, platform: "Facebook", comment_text: "Best purchase I’ve made in a long time!", sentiment: "positive" },
    { comment_id: 42, product_id: 2, customer_id: 42, platform: "Twitter", comment_text: "Very poor quality, wouldn’t recommend.", sentiment: "negative" },
    { comment_id: 43, product_id: 3, customer_id: 43, platform: "Instagram", comment_text: "Super happy with it! Works like a charm.", sentiment: "positive" },
    { comment_id: 44, product_id: 4, customer_id: 44, platform: "LinkedIn", comment_text: "It’s okay, but it could be improved.", sentiment: "neutral" },
    { comment_id: 45, product_id: 5, customer_id: 45, platform: "Snapchat", comment_text: "Amazing product! I’m so glad I bought it.", sentiment: "positive" }
]);

EOF
