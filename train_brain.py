import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# EXACTLY 36 items in each list to prevent the ValueError
data = {
    'description': [
        # Income (3)
        'Salary credited', 'Bonus payment', 'Freelance income',
        # Bills (5)
        'Mortgage payment', 'House Rent', 'Electricity bill', 'Water bill', 'Internet bill',
        # Investments (3)
        'Bought company shares', 'Mutual fund SIP', 'Dividend earnings',
        # Groceries (4)
        'Supermarket grocery', 'Blinkit order', 'Milk and bread', 'Fruits and vegetables',
        # Shopping (4)
        'Amazon shopping', 'Flipkart order', 'Purchased clothing', 'Home decor',
        # Food & Dining (4)
        'Enjoyed a meal at a diner', 'Restaurant dinner', 'Starbucks coffee', 'Lunch at fast food',
        # Travel (5)
        'Paid for car fuel', 'Petrol pump', 'Uber ride', 'Hotel accommodation', 'Flight tickets',
        # Lifestyle (3)
        'Paid for a haircut', 'Gym membership', 'Movie tickets',
        # Education (3)
        'Professional certification exam', 'Online course enrollment', 'Tuition fee payment',
        # Savings (2)
        'Transferred funds to savings', 'Emergency fund deposit'
    ],
    'category': [
        'Income', 'Income', 'Income',
        'Bills', 'Bills', 'Bills', 'Bills', 'Bills',
        'Investments', 'Investments', 'Investments',
        'Groceries', 'Groceries', 'Groceries', 'Groceries',
        'Shopping', 'Shopping', 'Shopping', 'Shopping',
        'Food & Dining', 'Food & Dining', 'Food & Dining', 'Food & Dining',
        'Travel', 'Travel', 'Travel', 'Travel', 'Travel',
        'Lifestyle', 'Lifestyle', 'Lifestyle',
        'Education', 'Education', 'Education',
        'Savings', 'Savings'
    ]
}

if len(data['description']) != len(data['category']):
    print(f"Error: List mismatch!")
else:
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['description'])
    y = df['category']
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'finance_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ AI Brain Trained Successfully!")
