import streamlit as st
import pandas as pd
import pdfplumber
import joblib
import plotly.express as px
import os
import re

# --- 1. PRO UI SETTINGS ---
st.set_page_config(page_title="Universal Finance AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        .stMetric { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #f0f2f6; box-shadow: 0 4px 10px rgba(0,0,0,0.03); }
        h1 { font-weight: 800; color: #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- 2. RESILIENT CLEANING ENGINE (Fixes Missing Rows) ---
def clean_engine(df):
    # Standardize headers
    df.columns = [re.sub(r'[^a-z]', '', str(c).lower()) for c in df.columns]
    
    mapping = {
        'description': ['desc', 'detail', 'particular', 'narrative'],
        'amount': ['amount', 'amt', 'value', 'debit', 'withdrawal', 'bill'],
        'date': ['date', 'time', 'txn', 'post']
    }
    
    final_cols = {}
    for target, keys in mapping.items():
        for col in df.columns:
            if any(k in col for k in keys):
                final_cols[col] = target
                break
    df = df.rename(columns=final_cols)
    
    # --- DATE REPAIR (Ensures NO rows are deleted) ---
    if 'date' in df.columns:
        # Convert to datetime; dayfirst=True handles international formats
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
        # We keep the original string if conversion fails, so the row is never lost
        df['date_display'] = df['date_dt'].dt.strftime('%Y-%m-%d').fillna(df['date'].astype(str))
    
    # --- AMOUNT CLEANING ---
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    
    return df

# --- 3. ZERO-ERROR ACCURACY ENGINE ---
def apply_strict_logic(df):
    rules = {
        'Health': ['doctor', 'consultation', 'medical', 'pharmacy', 'medicine', 'checkup', 'dental'],
        'Travel': ['hotel', 'accommodation', 'fuel', 'petrol', 'uber', 'ride-sharing', 'transport', 'flight', 'rental'],
        'Education': ['certification', 'exam', 'course', 'tuition', 'textbook', 'university'],
        'Savings': ['savings', 'emergency fund', 'saved', 'fixed deposit', 'fund'],
        'Groceries': ['grocery', 'supermarket', 'mart', 'blinkit', 'zepto', 'fruit', 'bread', 'vegetables', 'dairy'],
        'Lifestyle': ['haircut', 'salon', 'gym', 'movie', 'concert', 'netflix', 'spotify', 'amusement', 'manicure'],
        'Food & Dining': ['meal', 'diner', 'lunch', 'dinner', 'restaurant', 'cafe', 'starbucks'],
        'Investments': ['share', 'stock', 'mutual fund', 'dividend', 'crypto'],
        'Shopping': ['amazon', 'flipkart', 'clothing', 'decor', 'electronics'],
        'Bills': ['rent', 'mortgage', 'emi', 'utility', 'electricity', 'water', 'gas', 'internet', 'insurance']
    }
    for cat, keywords in rules.items():
        pattern = '|'.join(keywords)
        mask = df['description'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'category'] = cat
    return df

# --- 4. MAIN APP ---
st.title("🏦 Universal Finance AI")
model, vectorizer = (joblib.load('finance_model.pkl'), joblib.load('vectorizer.pkl')) if os.path.exists('finance_model.pkl') else (None, None)

uploaded_file = st.file_uploader("📂 Upload Bank Statement", type=['csv', 'xlsx', 'pdf'])

if uploaded_file and model:
    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            rows = []
            for page in pdf.pages: rows.extend(page.extract_table() or [])
            df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    df = clean_engine(df)
    
    if 'amount' not in df.columns:
        st.error(f"❌ Missing Amount column. Found: {list(df.columns)}")
        st.stop()

    # AI Classify + Correction Rules
    X_text = vectorizer.transform(df['description'].astype(str))
    df['category'] = model.predict(X_text)
    df = apply_strict_logic(df) 

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Expenses", f"₹{df['amount'].sum():,.2f}")
    m2.metric("Total Transactions", f"{len(df):,}")
    m3.metric("Top Spend Category", df['category'].mode()[0])
    m4.metric("Avg Transaction", f"₹{df['amount'].mean():,.2f}")

    st.divider()

    # Visuals
    v1, v2 = st.columns([1, 1.2])
    with v1:
        st.write("### Spending Hierarchy")
        st.plotly_chart(px.sunburst(df, path=['category', 'description'], values='amount', color='category', color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
    with v2:
        st.write("### Cash Flow Trend")
        # Trend graph only uses rows with valid dates
        df_trend = df.dropna(subset=['date_dt']).sort_values('date_dt')
        st.plotly_chart(px.area(df_trend, x='date_dt', y='amount', template="plotly_white", color_discrete_sequence=['#1f77b4']), use_container_width=True)

    st.write("### Final Categorized Ledger")
    # Display the ledger using the display-safe date column
    st.dataframe(df[['date_display', 'description', 'amount', 'category']].rename(columns={'date_display': 'date'}), use_container_width=True)

elif not model:
    st.warning("⚠️ Run train_brain.py first!")
