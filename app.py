import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="FootyScore AI", layout="centered")

# Locate historical_data.csv relative to app.py
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

    # 2. Convert goal columns to numeric values
    df[h_goals_col] = pd.to_numeric(df[h_goals_col], errors='coerce')
    df[a_goals_col] = pd.to_numeric(df[a_goals_col], errors='coerce')

    # 3. Calculate average goals scored and conceded
    home_scored = df[df['HomeTeam'] == home][h_goals_col].mean()
    home_conceded = df[df['HomeTeam'] == home][a_goals_col].mean()

    away_scored = df[df['AwayTeam'] == away][a_goals_col].mean()
    away_conceded = df[df['AwayTeam'] == away][h_goals_col].mean()

    # 4. League averages
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

    # 5. Poisson probability score matrix (0 to 5 goals)
    max_goals = 6
    p_matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p_matrix[h, a] = poisson.pmf(h, expected_home_goals) * poisson.pmf(a, expected_away_goals)

    # 6. Aggregate raw probabilities
    home_win_prob = float(np.sum(np.tril(p_matrix, -1)))
    draw_prob = float(np.sum(np.diag(p_matrix)))
    away_win_prob = float(np.sum(np.triu(p_matrix, 1)))

    # Normalize so probabilities sum cleanly to 100%
    total_p = home_win_prob + draw_prob + away_win_prob
    if total_p > 0:
        home_win_prob /= total_p
        draw_prob /= total_p
        away_win_prob /= total_p

    probs = {
        'Home Win': home_win_prob,
        'Draw': draw_prob,
        'Away Win': away_win_prob,
        'Expected Home Goals': expected_home_goals,
        'Expected Away Goals': expected_away_goals
    }

    return probs, p_matrix

# --- Main Interface ---
st.title("⚽ FootyScore AI")

df = load_default_data()

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

            st.divider()

            # --- Visualization 1: Score Matrix Heatmap ---
            st.write("### 📊 Correct Score Matrix Heatmap")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                p_matrix * 100, 
                annot=True, 
                fmt=".1f", 
                cmap="Blues", 
                xticklabels=range(6), 
                yticklabels=range(6), 
                ax=ax
            )
            ax.set_xlabel(f"{away_team} Goals")
            ax.set_ylabel(f"{home_team} Goals")
            ax.set_title("Probability of Exact Scores (%)")
            st.pyplot(fig)

            # --- Visualization 2: Top 5 Most Likely Scores ---
            st.write("### 🎯 Top 5 Most Likely Outcomes")
            
            # Find highest probabilities in the matrix
            scores = []
            for h in range(6):
                for a in range(6):
                    scores.append({'Score': f"{home_team} {h} - {a} {away_team}", 'Probability': p_matrix[h, a] * 100})
            
            top_scores_df = pd.DataFrame(scores).sort_values(by='Probability', ascending=False).head(5)
            st.bar_chart(top_scores_df.set_index('Score'))

    else:
        st.error("The dataset is missing 'HomeTeam' or 'AwayTeam' columns.")
else:
    st.error("Could not load `historical_data.csv`. Ensure it is in the repository.")
