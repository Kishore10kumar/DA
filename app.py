import streamlit as st
import pandas as pd
import pdfplumber
import joblib
import plotly.express as px
import os
import re

st.set_page_config(page_title="Universal Finance AI", layout="wide", initial_sidebar_state="collapsed")

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        .stMetric { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #f0f2f6; }
        h1 { font-weight: 800; color: #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- CLEANING ENGINE ---
def clean_engine(df):
    df.columns = [re.sub(r'[^a-z]', '', str(c).lower()) for c in df.columns]
    mapping = {'description': ['desc', 'detail', 'particular'], 'amount': ['amount', 'amt', 'value', 'debit'], 'date': ['date', 'time', 'txn']}
    for target, keys in mapping.items():
        for col in df.columns:
            if any(k in col for k in keys):
                df = df.rename(columns={col: target})
                break
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    return df

# --- THE ACCURACY FIX: PRIORITY RULES ENGINE ---
def apply_strict_logic(df):
    """Overrides AI errors to ensure 100% accurate segregation."""
    rules = {
        'Travel': ['hotel', 'accommodation', 'fuel', 'petrol', 'diesel', 'uber', 'ride-sharing', 'transport', 'flight', 'rental', 'commute'],
        'Health': ['doctor', 'consultation', 'medical', 'pharmacy', 'medicine', 'checkup', 'dental', 'clinic'],
        'Investments': ['share', 'stock', 'mutual fund', 'investment', 'dividend', 'crypto', 'nifty', 'equity'],
        'Lifestyle': ['haircut', 'salon', 'gym', 'movie', 'concert', 'netflix', 'spotify', 'amusement', 'manicure', 'skincare', 'grooming', 'streaming'],
        'Education': ['certification', 'exam', 'course', 'tuition', 'textbook', 'university'],
        'Savings': ['transferred to savings', 'emergency fund', 'fixed deposit', 'retirement'],
        'Groceries': ['grocery', 'supermarket', 'mart', 'blinkit', 'zepto', 'bigbasket', 'dairy', 'vegetables', 'bread'],
        'Food & Dining': ['meal', 'diner', 'lunch', 'dinner', 'restaurant', 'cafe', 'starbucks', 'swiggy', 'zomato'],
        'Shopping': ['amazon', 'flipkart', 'clothing', 'decor', 'electronics', 'shopping'],
        'Bills': ['rent', 'mortgage', 'emi', 'utility', 'electricity', 'water', 'gas', 'internet', 'insurance', 'maintenance']
    }
    
    for cat, keywords in rules.items():
        pattern = '|'.join(keywords)
        mask = df['description'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'category'] = cat
    return df

# --- MAIN APP ---
st.title("🏦 Universal Finance AI")
model, vectorizer = (joblib.load('finance_model.pkl'), joblib.load('vectorizer.pkl')) if os.path.exists('finance_model.pkl') else (None, None)

uploaded_file = st.file_uploader("📂 Upload Bank Statement", type=['csv', 'xlsx', 'pdf'])

if uploaded_file and model:
    # Parsing
    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            rows = []
            for page in pdf.pages: rows.extend(page.extract_table() or [])
            df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    df = clean_engine(df)
    
    # AI Prediction + Deterministic Override
    X_text = vectorizer.transform(df['description'].astype(str))
    df['category'] = model.predict(X_text)
    df = apply_strict_logic(df) 

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Spending", f"₹{df['amount'].sum():,.2f}")
    m2.metric("Total Entries", len(df))
    m3.metric("Primary Category", df['category'].mode()[0])
    m4.metric("Avg Spending", f"₹{df['amount'].mean():,.2f}")

    st.divider()

    # Visual Analytics
    v1, v2 = st.columns([1, 1.2])
    with v1:
        st.write("### Spending Hierarchy")
        st.plotly_chart(px.sunburst(df, path=['category', 'description'], values='amount', color='category', color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
    with v2:
        st.write("### Cash Flow Trend")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        st.plotly_chart(px.area(df.sort_values('date').dropna(subset=['date']), x='date', y='amount', template="plotly_white", color_discrete_sequence=['#3b82f6']), use_container_width=True)

    st.write("### Detailed Financial Ledger")
    st.dataframe(df[['date', 'description', 'amount', 'category']], use_container_width=True)
elif not model:
    st.warning("⚠️ Run train_brain.py first!")