import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AI Library Pro", layout="wide", page_icon="📚")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('google_books_dataset.csv')
    df['description'] = df['description'].fillna('')
    if 'is_available' not in df.columns:
        df['is_available'] = True
    return df

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'users' not in st.session_state:
    st.session_state.users = pd.DataFrame([
        {'user_id': 'A001', 'name': 'Admin User', 'role': 'Admin'},
        {'user_id': 'U001', 'name': 'Regular Reader', 'role': 'User'}
    ])

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
menu = ["Library Portal", "AI Recommendations", "Admin Dashboard"]
choice = st.sidebar.selectbox("Go to", menu)

# --- 1. Library Portal ---
if choice == "Library Portal":
    st.title("📚 Smart Library Portal")
    
    # Search UI
    st.subheader("🔍 Find a Book")
    search_q = st.text_input("Search by Title or Author", "")
    
    display_df = st.session_state.df.copy()
    if search_q:
        display_df = display_df[
            display_df['title'].str.contains(search_q, case=False, na=False) |
            display_df['authors'].str.contains(search_q, case=False, na=False)
        ]
    
    # Display Results with Status Icons
    display_df['Status'] = display_df['is_available'].map({True: '✅ Available', False: '❌ Borrowed'})
    st.dataframe(display_df[['title', 'authors', 'categories', 'Status']].head(20), use_container_width=True)

    st.divider()
    
    # Borrow/Return Operations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Borrow")
        avail_books = st.session_state.df[st.session_state.df['is_available'] == True]['title'].tolist()
        b_title = st.selectbox("Select book to borrow", [""] + avail_books)
        if st.button("Confirm Borrow") and b_title != "":
            idx = st.session_state.df[st.session_state.df['title'] == b_title].index[0]
            st.session_state.df.at[idx, 'is_available'] = False
            st.success(f"Enjoy reading '{b_title}'!")
            st.rerun()
            
    with col2:
        st.subheader("📥 Return")
        borrowed_books = st.session_state.df[st.session_state.df['is_available'] == False]['title'].tolist()
        if borrowed_books:
            r_title = st.selectbox("Select book to return", [""] + borrowed_books)
            if st.button("Process Return") and r_title != "":
                idx = st.session_state.df[st.session_state.df['title'] == r_title].index[0]
                st.session_state.df.at[idx, 'is_available'] = True
                st.success(f"'{r_title}' returned to catalog.")
                st.rerun()
        else:
            st.info("No books are currently out on loan.")

# --- 2. AI Recommendations ---
elif choice == "AI Recommendations":
    st.title("🤖 AI Recommendation Engine")
    st.write("Select a book you enjoyed, and our AI will find similar titles based on content analysis.")
    
    ref_book = st.selectbox("I liked reading...", st.session_state.df['title'].unique())
    
    if st.button("Generate Suggestions"):
        with st.spinner("Analyzing library content..."):
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(st.session_state.df['description'])
            
            try:
                idx = st.session_state.df[st.session_state.df['title'] == ref_book].index[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                sim_indices = cosine_sim.argsort()[-6:-1][::-1]
                recs = st.session_state.df.iloc[sim_indices]
                
                st.subheader(f"Because you liked **{ref_book}**:")
                for _, row in recs.iterrows():
                    st.write(f"📖 **{row['title']}** by {row['authors']}")
                    st.caption(f"Category: {row['categories']}")
            except Exception as e:
                st.error("Could not generate recommendations for this title.")

# --- 3. Admin Dashboard ---
elif choice == "Admin Dashboard":
    st.title("🛠 Admin Management Center")
    admin_pwd = st.sidebar.text_input("Admin Access Key", type="password")
    
    if admin_pwd == "admin123":
        t1, t2 = st.tabs(["Inventory", "Users"])
        
        with t1:
            st.subheader("System Inventory")
            st.write(f"Total Books: {len(st.session_state.df)}")
            st.dataframe(st.session_state.df[['book_id', 'title', 'is_available']].head(50))
            
            if st.button("Export CSV"):
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Inventory", data=csv, file_name="inventory_export.csv", mime="text/csv")
        
        with t2:
            st.subheader("Registered Patrons")
            st.table(st.session_state.users)
    else:
        st.warning("Please enter the admin password in the sidebar to unlock management tools.")
