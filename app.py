import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# Set page layout
st.set_page_config(page_title="AI Match Predictor", layout="wide")
st.title("⚽ AI Match Predictor")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("historical_data.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Could not load 'historical_data.csv'. Make sure the CSV file is uploaded to your GitHub repository.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("Match Setup")
teams = sorted(df['HomeTeam'].unique()) if 'HomeTeam' in df.columns else []

if not teams:
    st.error("Column 'HomeTeam' not found in dataset. Please check your CSV column names.")
    st.stop()

home_team = st.sidebar.selectbox("Home Team", teams, index=0)
away_team = st.sidebar.selectbox("Away Team", teams, index=min(1, len(teams)-1))

# 3. Model Logic (Poisson Expectation Engine)
def predict_match(df, home, away):
    # Overall averages
    avg_home_goals = df['FTHG'].mean() if 'FTHG' in df.columns else df['HG'].mean()
    avg_away_goals = df['FTAG'].mean() if 'FTAG' in df.columns else df['AG'].mean()
    
    # Team stats
    home_scored = df[df['HomeTeam'] == home]['FTHG' if 'FTHG' in df.columns else 'HG'].mean()
    home_conceded = df[df['HomeTeam'] == home]['FTAG' if 'FTAG' in df.columns else 'AG'].mean()
    
    away_scored = df[df['AwayTeam'] == away]['FTAG' if 'FTAG' in df.columns else 'AG'].mean()
    away_conceded = df[df['AwayTeam'] == away]['HomeTeam' if 'HomeTeam' in df.columns else 'HG'].mean()
    
    # Strengths
    home_att = home_scored / avg_home_goals if avg_home_goals else 1.0
    home_def = home_conceded / avg_away_goals if avg_away_goals else 1.0
    away_att = away_scored / avg_away_goals if avg_away_goals else 1.0
    away_def = away_conceded / avg_home_goals if avg_home_goals else 1.0
    
    # Expected goals (lambda)
    exp_home_goals = home_att * away_def * avg_home_goals
    exp_away_goals = away_att * home_def * avg_away_goals
    
    # Score matrix (0 to 5 goals)
    max_goals = 6
    p_matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p_matrix[h, a] = poisson.pmf(h, exp_home_goals) * poisson.pmf(a, exp_away_goals)
            
    p_matrix /= p_matrix.sum() # Normalize
    
    home_win = np.sum(np.tril(p_matrix, -1))
    draw = np.sum(np.diag(p_matrix))
    away_win = np.sum(np.triu(p_matrix, 1))
    
    return [home_win, draw, away_win], p_matrix

# 4. Main Display
if st.sidebar.button("Generate Prediction", type="primary"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        probs, p_matrix = predict_match(df, home_team, away_team)
        
        st.subheader(f"{home_team} vs {away_team}")
        
        # Key Metrics Display
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{probs[0]*100:.1f}%")
        col2.metric("Draw", f"{probs[1]*100:.1f}%")
        col3.metric("Away Win", f"{probs[2]*100:.1f}%")
        
        # Visualization Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar Chart
        categories = ['Home Win', 'Draw', 'Away Win']
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        bars = ax1.barh(categories, probs, color=colors)
        ax1.set_xlim(0, 1.0)
        ax1.set_title("Match Probabilities", fontsize=14)
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%", va='center')
        ax1.invert_yaxis()
        
        # Heatmap
        sns.heatmap(p_matrix * 100, annot=True, fmt=".1f", cmap="YlGnBu", 
                    xticklabels=range(6), yticklabels=range(6), ax=ax2, cbar=False)
        ax2.set_title("Exact Scoreline Probabilities (%)", fontsize=14)
        ax2.set_xlabel(f"{away_team} Goals")
        ax2.set_ylabel(f"{home_team} Goals")
        
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("Select teams in the sidebar and click 'Generate Prediction' to run the model.")
