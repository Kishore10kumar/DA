import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Expanded and Balanced Dataset (Exactly 5 examples per category for stability)
data = {
    'description': [
        'Salary credited to account', 'Monthly bonus', 'Freelance payment', 'Dividend income', 'Cashback received',
        'House rent payment', 'Electricity bill', 'Water utility bill', 'Internet subscription', 'Insurance premium',
        'Bought company shares', 'Mutual fund SIP', 'Stock investment', 'Equity purchase', 'Crypto investment',
        'Transferred to savings', 'Emergency fund deposit', 'Fixed deposit plan', 'Savings account interest', 'Retirement fund',
        'Weekly grocery shopping', 'Supermarket bill', 'Blinkit order', 'Zepto groceries', 'Milk and vegetables',
        'Dinner at restaurant', 'Starbucks coffee', 'Lunch at diner', 'Swiggy food order', 'Zomato delivery',
        'Paid for car fuel', 'Uber ride', 'Flight tickets', 'Hotel accommodation', 'Public transport card',
        'Amazon shopping', 'Flipkart order', 'Purchased clothing', 'Electronics purchase', 'Home decor items',
        'Paid for a haircut', 'Gym membership', 'Movie tickets', 'Netflix subscription', 'Spotify premium',
        'Doctor consultation fee', 'Pharmacy medicine bill', 'Dental treatment', 'Medical checkup', 'Health insurance',
        'Certification exam fee', 'Online course payment', 'Tuition fee', 'University textbooks', 'Education loan'
    ],
    'category': [
        'Income', 'Income', 'Income', 'Income', 'Income',
        'Bills', 'Bills', 'Bills', 'Bills', 'Bills',
        'Investments', 'Investments', 'Investments', 'Investments', 'Investments',
        'Savings', 'Savings', 'Savings', 'Savings', 'Savings',
        'Groceries', 'Groceries', 'Groceries', 'Groceries', 'Groceries',
        'Food & Dining', 'Food & Dining', 'Food & Dining', 'Food & Dining', 'Food & Dining',
        'Travel', 'Travel', 'Travel', 'Travel', 'Travel',
        'Shopping', 'Shopping', 'Shopping', 'Shopping', 'Shopping',
        'Lifestyle', 'Lifestyle', 'Lifestyle', 'Lifestyle', 'Lifestyle',
        'Health', 'Health', 'Health', 'Health', 'Health',
        'Education', 'Education', 'Education', 'Education', 'Education'
    ]
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['description'])
y = df['category']

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

joblib.dump(model, 'finance_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ High-Precision Model Ready!")