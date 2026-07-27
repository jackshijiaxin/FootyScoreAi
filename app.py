import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FootyScore AI", layout="centered")

# --- API & File Configuration ---
API_KEY = st.secrets.get("FOOTBALL_API_KEY", "YOUR_API_KEY_HERE")
API_HOST = "v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "historical_data.csv")
PLAYERS_CSV_PATH = os.path.join(BASE_DIR, "players_data.csv")

# Team name translation dictionary (football-data.co.uk -> API-Football)
TEAM_NAME_MAP = {
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Paris SG": "Paris Saint Germain",
    "Leverkusen": "Bayer Leverkusen",
    "Inter": "Inter Milan",
    "Bayern Munich": "Bayern Munich",
    "Spurs": "Tottenham Hotspur"
}

LEAGUES_CONFIG = {
    "English Premier League": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
        "api_league_id": 39
    },
    "Spanish La Liga": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
        "api_league_id": 140
    },
    "Italian Serie A": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
        "api_league_id": 135
    },
    "German Bundesliga": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
        "api_league_id": 78
    },
    "French Ligue 1": {
        "match_url": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
        "api_league_id": 61
    }
}

# --- Data Loaders ---
@st.cache_data(ttl="1d")
def load_match_data(league_name):
    """Loads historical team match data for the selected league."""
    url = LEAGUES_CONFIG[league_name]["match_url"]
    try:
        df_league = pd.read_csv(url)
        if df_league is not None and not df_league.empty:
            return df_league
    except Exception:
        pass

    if os.path.exists(CSV_FILE_PATH):
        return pd.read_csv(CSV_FILE_PATH)
    return None

@st.cache_data(ttl=3600)
def load_players_csv():
    """Reads the pre-built player dataset generated daily by GitHub Actions."""
    if os.path.exists(PLAYERS_CSV_PATH):
        return pd.read_csv(PLAYERS_CSV_PATH)
    return None

@st.cache_data(ttl=86400)
def fetch_api_team_players(team_name, league_id):
    """Fallback: Fetches full team squad directly from API-Football if CSV missing."""
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        return None

    try:
        search_name = TEAM_NAME_MAP.get(team_name, team_name)
        search_url = f"https://{API_HOST}/teams"
        params_team = {"search": search_name}
        res_team = requests.get(search_url, headers=HEADERS, params=params_team, timeout=10).json()

        if not res_team.get("response"):
            return None

        team_id = res_team["response"][0]["team"]["id"]
        player_list = []

        # Loop through pages to get full squad
        for page in range(1, 3):
            players_url = f"https://{API_HOST}/players"
            params_players = {"team": team_id, "season": 2024, "league": league_id, "page": page}
            res_players = requests.get(players_url, headers=HEADERS, params=params_players, timeout=10).json()

            items = res_players.get("response", [])
            if not items:
                break

            for item in items:
                player_info = item["player"]
                stats = item["statistics"][0]

                games_played = stats["games"].get("appearences") or 0
                position = stats["games"].get("position") or "Attacker"

                if games_played < 5:
                    continue

                raw_rating = stats["games"].get("rating")
                rating = float(raw_rating) if raw_rating else 6.5
                goals = stats["goals"].get("total") or 0
                assists = stats["goals"].get("assists") or 0

                player_list.append({
                    "Team": team_name,
                    "Player": player_info["name"],
                    "Position": position,
                    "Goals": goals,
                    "Assists": assists,
                    "Rating": round(rating, 2)
                })

        if player_list:
            return pd.DataFrame(player_list).drop_duplicates(subset=["Player"])

    except Exception:
        pass

    return None

# --- Match Prediction Logic ---
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

    return h2h[display_cols].head(2)

def display_player_predictions(team_name, league_id, expected_team_goals):
    """Displays key outfield players, probabilities, and star player."""
    team_df = None

    # 1. Attempt to load from daily CSV first
    players_csv = load_players_csv()
    if players_csv is not None and not players_csv.empty:
        mapped_name = TEAM_NAME_MAP.get(team_name, team_name)
        matched = players_csv[
            (players_csv["Team"].str.lower() == team_name.lower()) |
            (players_csv["Team"].str.lower() == mapped_name.lower()) |
            (players_csv.get("OfficialTeam", pd.Series()).str.lower() == mapped_name.lower())
        ]
        if not matched.empty:
            team_df = matched.copy()

    # 2. Fallback to live API if CSV isn't available
    if team_df is None or team_df.empty:
        team_df = fetch_api_team_players(team_name, league_id)

    if team_df is not None and not team_df.empty:
        # Strictly exclude Goalkeepers from star player & target calculations
        outfield_df = team_df[team_df["Position"] != "Goalkeeper"].copy()
        if outfield_df.empty:
            outfield_df = team_df.copy()

        total_goals = outfield_df["Goals"].sum()
        total_assists = outfield_df["Assists"].sum()

        if total_goals > 0:
            outfield_df['Scoring Prob (%)'] = (1 - np.exp(-(outfield_df['Goals'] / total_goals) * expected_team_goals)) * 100
        else:
            outfield_df['Scoring Prob (%)'] = 0.0

        if total_assists > 0:
            outfield_df['Assist Prob (%)'] = (1 - np.exp(-(outfield_df['Assists'] / total_assists) * expected_team_goals)) * 100
        else:
            outfield_df['Assist Prob (%)'] = 0.0

        # Select top outfield Star Player by rating
        star_player = outfield_df.sort_values(by="Rating", ascending=False).iloc[0]
        st.markdown(f"⭐ **Star Player:** **{star_player['Player']}** ({star_player['Position']} - Rating: **{star_player['Rating']}/10**)")

        # Display Top 5 outfield goal targets
        outfield_df["Rank_Score"] = outfield_df["Goals"] * 2 + outfield_df["Assists"] + outfield_df["Rating"]
        top_targets = outfield_df.sort_values(by="Rank_Score", ascending=False).head(5)

        st.dataframe(
            top_targets[["Player", "Position", "Scoring Prob (%)", "Assist Prob (%)"]].style.format({'Scoring Prob (%)': '{:.1f}%', 'Assist Prob (%)': '{:.1f}%'}),
            hide_index=True
        )
        return

    st.info("Player data updating... Ensure GitHub Action workflow has completed or API Key is set.")


# --- Streamlit UI ---
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

        # Head to Head
        st.write("### 📜 Head-to-Head (Last 2 Meetings)")
        h2h_df = get_h2h_matches(df, home_team, away_team)
        if h2h_df is not None and not h2h_df.empty:
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)
        else:
            st.write("No direct Head-to-Head matches found in this dataset.")

        st.divider()

        # Key Player Predictions & Star Players
        st.write("### 👟 Key Player Predictions & Star Players")
        league_id = LEAGUES_CONFIG[selected_league]["api_league_id"]

        with st.spinner("Loading squad ratings and predictions..."):
            col_h, col_a = st.columns(2)
            with col_h:
                st.write(f"**{home_team} Targets**")
                display_player_predictions(home_team, league_id, probs['Expected Home Goals'])

            with col_a:
                st.write(f"**{away_team} Targets**")
                display_player_predictions(away_team, league_id, probs['Expected Away Goals'])

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
