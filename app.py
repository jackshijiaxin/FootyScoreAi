import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import soccerdata as sd

st.set_page_config(page_title="FootyScore AI", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "historical_data.csv")

LEAGUES_CONFIG = {
    "English Premier League": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "sd_code": "ENG-Premier League"
    },
    "Spanish La Liga": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "sd_code": "ESP-La Liga"
    },
    "Italian Serie A": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "sd_code": "ITA-Serie A"
    },
    "German Bundesliga": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "sd_code": "GER-Bundesliga"
    },
    "French Ligue 1": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "sd_code": "FRA-Ligue 1"
    }
}

@st.cache_data(ttl="1d")
def load_match_data(league_name):
    """Loads historical team match data."""
    if os.path.exists(CSV_FILE_PATH):
        local_df = pd.read_csv(CSV_FILE_PATH)
        if 'League' in local_df.columns:
            filtered = local_df[local_df['League'] == league_name]
            if not filtered.empty:
                return filtered
        else:
            return local_df
    
    url = LEAGUES_CONFIG[league_name]["match_url"]
    return pd.read_csv(url)

@st.cache_data(ttl=86400)
def load_player_stats(sd_code):
    """Fetches FBref player stats via soccerdata for top 5 leagues."""
    try:
        fbref = sd.FBref(leagues=sd_code, seasons="2425")
        df_players = fbref.read_player_season_stats(stat_type="standard")
        if df_players is not None and not df_players.empty:
            return df_players.reset_index()
    except Exception:
        pass
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

    max_score_idx = np.unravel_index(np.argmax(p_matrix, axis=None), p_matrix.shape)
    top_home_g, top_away_g = max_score_idx[0], max_score_idx[1]
    top_score_prob = p_matrix[top_home_g, top_away_g] * 100

    return {
        'Home Win': home_win_prob,
        'Draw': draw_prob,
        'Away Win': away_win_prob,
        'Expected Home Goals': expected_home_goals,
        'Expected Away Goals': expected_away_goals,
        'Top Score': f"{top_home_g} - {top_away_g}",
        'Top Score Prob': top_score_prob
    }, p_matrix

def get_h2h_matches(df, team1, team2):
    """Filters data for the last 2 direct encounters between team1 and team2."""
    h2h = df[
        ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
        ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
    ].copy()

    if h2h.empty:
        return None

    h_col = 'FTHG' if 'FTHG' in h2h.columns else 'HG'
    a_col = 'FTAG' if 'FTAG' in h2h.columns else 'AG'

    if 'Date' in h2h.columns:
        h2h['Date'] = pd.to_datetime(h2h['Date'], dayfirst=True, errors='coerce')
        h2h = h2h.sort_values(by='Date', ascending=False)
        h2h['Date'] = h2h['Date'].dt.strftime('%Y-%m-%d')

    h2h['Score'] = h2h[h_col].astype(str) + ' - ' + h2h[a_col].astype(str)
    display_cols = [c for c in ['Date', 'HomeTeam', 'Score', 'AwayTeam'] if c in h2h.columns]

    # Return only the last 2 matches
    return h2h[display_cols].head(2)

def display_player_predictions(player_df, team_name, expected_team_goals):
    """Calculates individual player scoring/assisting chances and identifies Star Player."""
    if player_df is not None and not player_df.empty:
        try:
            df = player_df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(c) for c in col if str(c) != '']).strip() for col in df.columns]

            team_col = [c for c in df.columns if 'team' in c.lower() or 'squad' in c.lower()][0]
            player_col = [c for c in df.columns if 'player' in c.lower()][0]
            xg_col = [c for c in df.columns if 'xg' in c.lower() and 'per' not in c.lower() and 'assist' not in c.lower()][0]
            xa_col = [c for c in df.columns if 'xg_assist' in c.lower() or 'xa' in c.lower() or 'ast' in c.lower()][0]

            team_players = df[df[team_col].astype(str).str.contains(team_name, case=False, na=False)].copy()

            if not team_players.empty:
                team_players[xg_col] = pd.to_numeric(team_players[xg_col], errors='coerce').fillna(0)
                team_players[xa_col] = pd.to_numeric(team_players[xa_col], errors='coerce').fillna(0)

                total_team_xg = team_players[xg_col].sum()
                if total_team_xg > 0:
                    team_players['Match_xG'] = (team_players[xg_col] / total_team_xg) * expected_team_goals
                    team_players['Goal Prob (%)'] = (1 - np.exp(-team_players['Match_xG'])) * 100
                else:
                    team_players['Goal Prob (%)'] = 0.0

                total_team_xa = team_players[xa_col].sum()
                if total_team_xa > 0:
                    team_players['Match_xA'] = (team_players[xa_col] / total_team_xa) * expected_team_goals
                    team_players['Assist Prob (%)'] = (1 - np.exp(-team_players['Match_xA'])) * 100
                else:
                    team_players['Assist Prob (%)'] = 0.0

                # Determine Star Player (highest combined goal involvement)
                team_players['Star_Score'] = team_players['Goal Prob (%)'] + (team_players['Assist Prob (%)'] * 0.5)
                star_row = team_players.sort_values(by='Star_Score', ascending=False).iloc[0]
                star_name = star_row[player_col]
                star_goal_prob = star_row['Goal Prob (%)']

                st.markdown(f"⭐ **Star Player:** **{star_name}** ({star_goal_prob:.1f}% goal prob)")

                top_scorers = team_players[[player_col, 'Goal Prob (%)', 'Assist Prob (%)']].sort_values(by='Goal Prob (%)', ascending=False).head(5)
                top_scorers.columns = ['Player', 'Scoring Prob (%)', 'Assist Prob (%)']
                
                st.dataframe(top_scorers.style.format({'Scoring Prob (%)': '{:.1f}%', 'Assist Prob (%)': '{:.1f}%'}), hide_index=True)
                return
        except Exception:
            pass

    # Fallback Star Player & Targets estimation
    st.markdown("⭐ **Star Player:** **Main Forward / Striker**")
    st.caption("*(Estimated distribution based on expected team goals)*")
    fallback_data = [
        {"Player": "Main Forward / Striker", "Scoring Prob (%)": min(expected_team_goals * 38.0, 85.0), "Assist Prob (%)": min(expected_team_goals * 18.0, 50.0)},
        {"Player": "Left Winger / Inside Forward", "Scoring Prob (%)": min(expected_team_goals * 26.0, 70.0), "Assist Prob (%)": min(expected_team_goals * 22.0, 55.0)},
        {"Player": "Right Winger", "Scoring Prob (%)": min(expected_team_goals * 24.0, 68.0), "Assist Prob (%)": min(expected_team_goals * 25.0, 60.0)},
        {"Player": "Attacking Midfielder", "Scoring Prob (%)": min(expected_team_goals * 16.0, 45.0), "Assist Prob (%)": min(expected_team_goals * 30.0, 65.0)},
        {"Player": "Central Midfielder", "Scoring Prob (%)": min(expected_team_goals * 10.0, 30.0), "Assist Prob (%)": min(expected_team_goals * 15.0, 40.0)},
    ]
    st.dataframe(pd.DataFrame(fallback_data).style.format({'Scoring Prob (%)': '{:.1f}%', 'Assist Prob (%)': '{:.1f}%'}), hide_index=True)


# --- Interface ---
st.title("⚽ FootyScore AI")

selected_league = st.selectbox("Select League", list(LEAGUES_CONFIG.keys()))
df = load_match_data(selected_league)

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

        st.success(f"🎯 **Most Likely Score:** {home_team} **{probs['Top Score']}** {away_team} ({probs['Top Score Prob']:.1f}% probability)")
        st.info(f"**Expected Goals:** {home_team} ({probs['Expected Home Goals']:.2f}) - ({probs['Expected Away Goals']:.2f}) {away_team}")

        st.divider()

        # Head to Head (Last 2 Meetings)
        st.write("### 📜 Head-to-Head (Last 2 Meetings)")
        h2h_df = get_h2h_matches(df, home_team, away_team)
        if h2h_df is not None and not h2h_df.empty:
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)
        else:
            st.write("No direct Head-to-Head matches found in this dataset.")

        st.divider()

        # Player Predictions & Star Player Highlights
        st.write("### 👟 Key Player Predictions")
        sd_code = LEAGUES_CONFIG[selected_league]["sd_code"]
        
        with st.spinner("Fetching player data..."):
            player_df = load_player_stats(sd_code)

        col_h, col_a = st.columns(2)
        with col_h:
            st.write(f"**{home_team} Targets**")
            display_player_predictions(player_df, home_team, probs['Expected Home Goals'])
            
        with col_a:
            st.write(f"**{away_team} Targets**")
            display_player_predictions(player_df, away_team, probs['Expected Away Goals'])

        st.divider()

        # Visualizations
        st.write("### 📊 Correct Score Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(p_matrix * 100, annot=True, fmt=".1f", cmap="Blues", xticklabels=range(6), yticklabels=range(6), ax=ax)
        ax.set_xlabel(f"{away_team} Goals")
        ax.set_ylabel(f"{home_team} Goals")
        st.pyplot(fig)

        st.write("### 🎯 Top 5 Most Likely Outcomes")
        scores = []
        for h in range(6):
            for a in range(6):
                scores.append({'Score': f"{home_team} {h} - {a} {away_team}", 'Probability': p_matrix[h, a] * 100})
        
        top_scores_df = pd.DataFrame(scores).sort_values(by='Probability', ascending=False).head(5)
        st.bar_chart(top_scores_df.set_index('Score'))

else:
    st.error("Could not load match data for the selected league.")
