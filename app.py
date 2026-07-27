import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Page setup
st.set_page_config(page_title="FootyScore AI", layout="centered")

# Determine exact path to historical_data.csv relative to app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "historical_data.csv")

@st.cache_data
def load_default_data():
    if os.path.exists(CSV_FILE_PATH):
        return pd.read_csv(CSV_FILE_PATH)
    return None

def predict_match(df, home, away):
    # 1. Identify goal column names dynamically
    h_goals_col = 'FTHG' if 'FTHG' in df.columns else 'HG'
    a_goals_col = 'FTAG' if 'FTAG' in df.columns else 'AG'

    # 2. Clean numeric data
    df[h_goals_col] = pd.to_numeric(df[h_goals_col], errors='coerce')
    df[a_goals_col] = pd.to_numeric(df[a_goals_col], errors='coerce')

    # 3. Calculate averages
    home_scored = df[df['HomeTeam'] == home][h_goals_col].mean()
    home_conceded = df[df['HomeTeam'] == home][a_goals_col].mean()

    away_scored = df[df['AwayTeam'] == away][a_goals_col].mean()
    away_conceded = df[df['AwayTeam'] == away][h_goals_col].mean()

    avg_home_scored = df[h_goals_col].mean()
    avg_away_scored = df[a_goals_col].mean()

    if pd.isna(home_scored) or pd.isna(away_conceded) or avg_home_scored == 0:
        expected_home_goals = 1.0
    else:
        home_attack = home_scored / avg_home_scored
        away_defense = away_conceded / avg_home_scored
        expected_home_goals = home_attack * away_defense * avg_home_scored

    if pd.isna(away_scored) or pd.isna(home_conceded) or avg_away_scored == 0:
        expected_away_goals = 1.0
    else:
        away_attack = away_scored / avg_away_scored
        home_defense = home_conceded / avg_away_scored
        expected_away_goals = away_attack * home_defense * avg_away_scored

    # 4. Poisson matrix calculation
    max_goals = 6
    p_matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p_matrix[h, a] = poisson.pmf(h, expected_home_goals) * poisson.pmf(a, expected_away_goals)

    probs = {
        'Home Win': float(np.sum(np.tril(p_matrix, -1))),
        'Draw': float(np.sum(np.diag(p_matrix))),
        'Away Win': float(np.sum(np.triu(p_matrix, 1))),
        'Expected Home Goals': expected_home_goals,
        'Expected Away Goals': expected_away_goals
    }

    return probs, p_matrix

# --- Main App Interface ---
st.title("⚽ FootyScore AI")

df = load_default_data()

# Optional sidebar uploader if you ever want to override historical_data.csv
uploaded_file = st.sidebar.file_uploader("Upload custom CSV (Optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if df is not None:
    required_cols = ['HomeTeam', 'AwayTeam']
    if all(col in df.columns for col in required_cols):
        teams = sorted(df['HomeTeam'].dropna().unique())

        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Select Home Team", teams)
        with col2:
            away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])

        if st.button("Predict Match", type="primary"):
            probs, p_matrix = predict_match(df, home_team, away_team)

            st.subheader(f"{home_team} vs {away_team}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Home Win", f"{probs['Home Win'] * 100:.1f}%")
            c2.metric("Draw", f"{probs['Draw'] * 100:.1f}%")
            c3.metric("Away Win", f"{probs['Away Win'] * 100:.1f}%")

            st.info(f"**Expected Goals:** {home_team} ({probs['Expected Home Goals']:.2f}) - ({probs['Expected Away Goals']:.2f}) {away_team}")
    else:
        st.error("The dataset missing 'HomeTeam' or 'AwayTeam' columns.")
else:
    st.error("Could not load `historical_data.csv`. Please ensure it is pushed to the root directory on GitHub.")
