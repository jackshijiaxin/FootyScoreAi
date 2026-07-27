import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FootyScore AI", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "historical_data.csv")

# Direct links to top 5 league data from Football-Data.co.uk
LEAGUE_URLS = {
    "English Premier League": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "Spanish La Liga": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "Italian Serie A": "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    "German Bundesliga": "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    "French Ligue 1": "https://www.football-data.co.uk/mmz4281/2324/F1.csv"
}

@st.cache_data
def load_data(league_name):
    # Check if local file exists and has a 'League' column
    if os.path.exists(CSV_FILE_PATH):
        local_df = pd.read_csv(CSV_FILE_PATH)
        if 'League' in local_df.columns:
            return local_df[local_df['League'] == league_name]
    
    # Fallback: Download league directly from web
    url = LEAGUE_URLS.get(league_name)
    if url:
        return pd.read_csv(url)
    return None

def predict_match(df, home, away):
    h_goals_col = 'FTHG' if 'FTHG' in df.columns else 'HG'
    a_goals_col = 'FTAG' if 'FTAG' in df.columns else 'AG'

    df[h_goals_col] = pd.to_numeric(df[h_goals_col], errors='coerce')
    df[a_goals_col] = pd.to_numeric(df[a_goals_col], errors='coerce')

    home_scored = df[df['HomeTeam'] == home][h_goals_col].mean()
    home_conceded = df[df['HomeTeam'] == home][a_goals_col].mean()
    away_scored = df[df['AwayTeam'] == away][a_goals_col].mean()
    away_conceded = df[df['AwayTeam'] == away][h_goals_col].mean()

    avg_home_scored = df[h_goals_col].mean()
    avg_away_scored = df[a_goals_col].mean()

    expected_home_goals = (home_scored / avg_home_scored) * (away_conceded / avg_home_scored) * avg_home_scored
    expected_away_goals = (away_scored / avg_away_scored) * (home_conceded / avg_away_scored) * avg_away_scored

    max_goals = 6
    p_matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p_matrix[h, a] = poisson.pmf(h, expected_home_goals) * poisson.pmf(a, expected_away_goals)

    home_win_prob = float(np.sum(np.tril(p_matrix, -1)))
    draw_prob = float(np.sum(np.diag(p_matrix)))
    away_win_prob = float(np.sum(np.triu(p_matrix, 1)))

    total_p = home_win_prob + draw_prob + away_win_prob
    if total_p > 0:
        home_win_prob /= total_p
        draw_prob /= total_p
        away_win_prob /= total_p

    return {
        'Home Win': home_win_prob,
        'Draw': draw_prob,
        'Away Win': away_win_prob,
        'Expected Home Goals': expected_home_goals,
        'Expected Away Goals': expected_away_goals
    }, p_matrix

# --- App Interface ---
st.title("⚽ FootyScore AI")

# League Selector
selected_league = st.selectbox("Select League", list(LEAGUE_URLS.keys()))
df = load_data(selected_league)

if df is not None and all(c in df.columns for c in ['HomeTeam', 'AwayTeam']):
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

        # Score Matrix Heatmap
        st.write("### 📊 Correct Score Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(p_matrix * 100, annot=True, fmt=".1f", cmap="Blues", xticklabels=range(6), yticklabels=range(6), ax=ax)
        ax.set_xlabel(f"{away_team} Goals")
        ax.set_ylabel(f"{home_team} Goals")
        st.pyplot(fig)

        # Top Outcomes
        st.write("### 🎯 Top 5 Most Likely Outcomes")
        scores = []
        for h in range(6):
            for a in range(6):
                scores.append({'Score': f"{home_team} {h} - {a} {away_team}", 'Probability': p_matrix[h, a] * 100})
        
        top_scores_df = pd.DataFrame(scores).sort_values(by='Probability', ascending=False).head(5)
        st.bar_chart(top_scores_df.set_index('Score'))

else:
    st.error("Could not load match data for the selected league.")
