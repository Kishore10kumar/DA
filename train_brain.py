import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    'description': [
        'Salary credited', 'Bonus payment', 'Freelance income',
        'Mortgage payment', 'House Rent', 'Electricity bill', 'Water bill', 'Internet bill',
        'Bought company shares', 'Mutual fund SIP', 'Dividend earnings',
        'Supermarket grocery', 'Blinkit order', 'Milk and bread', 'Fruits and vegetables',
        'Amazon shopping', 'Flipkart order', 'Purchased clothing', 'Home decor',
        'Enjoyed a meal at a diner', 'Restaurant dinner', 'Starbucks coffee',
        'Paid for car fuel', 'Petrol pump', 'Uber ride', 'Hotel accommodation', 'Flight tickets',
        'Paid for a haircut', 'Gym membership', 'Movie tickets', 'Netflix subscription',
        'Professional certification exam', 'Online course enrollment', 'Tuition fee payment',
        'Transferred funds to savings', 'Emergency fund deposit', 'Fixed deposit plan',
        'Doctor consultation fee', 'Medical checkup', 'Dental treatment', 'Pharmacy bill'
    ],
    'category': [
        'Income', 'Income', 'Income',
        'Bills', 'Bills', 'Bills', 'Bills', 'Bills',
        'Investments', 'Investments', 'Investments',
        'Groceries', 'Groceries', 'Groceries', 'Groceries',
        'Shopping', 'Shopping', 'Shopping', 'Shopping',
        'Food & Dining', 'Food & Dining', 'Food & Dining',
        'Travel', 'Travel', 'Travel', 'Travel', 'Travel',
        'Lifestyle', 'Lifestyle', 'Lifestyle', 'Lifestyle',
        'Education', 'Education', 'Education',
        'Savings', 'Savings', 'Savings',
        'Health', 'Health', 'Health', 'Health'
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
print("✅ AI Brain Ready!")
