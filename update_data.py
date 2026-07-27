import os
import requests
import pandas as pd

API_KEY = os.getenv("FOOTBALL_API_KEY")
API_HOST = "v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}

# Maps short team names from football-data.co.uk to API-Football official names
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

# Add key league IDs (e.g., EPL = 39, La Liga = 140)
LEAGUES = [39, 140] 

def fetch_squad_data():
    if not API_KEY:
        print("No API Key found in environment variables.")
        return

    all_players = []

    for league_id in LEAGUES:
        # Get standings to grab all team IDs in the league
        url = f"https://{API_HOST}/standings"
        params = {"season": 2024, "league": league_id}
        res = requests.get(url, headers=HEADERS, params=params).json()

        try:
            teams = res["response"][0]["league"]["standings"][0]
        except (KeyError, IndexError):
            continue

        for team_entry in teams:
            team_id = team_entry["team"]["id"]
            api_team_name = team_entry["team"]["name"]

            # Reverse map or use API team name
            mapped_team_name = api_team_name
            for short_name, official_name in TEAM_NAME_MAP.items():
                if official_name.lower() == api_team_name.lower():
                    mapped_team_name = short_name
                    break

            # Fetch players for this team
            p_url = f"https://{API_HOST}/players"
            p_params = {"team": team_id, "season": 2024, "league": league_id, "page": 1}
            p_res = requests.get(p_url, headers=HEADERS, params=p_params).json()

            for item in p_res.get("response", []):
                p_info = item["player"]
                stats = item["statistics"][0]

                position = stats["games"].get("position") or "Attacker"
                games = stats["games"].get("appearences") or 0

                # Skip goalkeepers & players with few games
                if position == "Goalkeeper" or games < 5:
                    continue

                raw_rating = stats["games"].get("rating")
                rating = float(raw_rating) if raw_rating else 6.5
                goals = stats["goals"].get("total") or 0
                assists = stats["goals"].get("assists") or 0

                all_players.append({
                    "Team": mapped_team_name,
                    "OfficialTeam": api_team_name,
                    "Player": p_info["name"],
                    "Position": position,
                    "Goals": goals,
                    "Assists": assists,
                    "Rating": round(rating, 2)
                })

    if all_players:
        df = pd.DataFrame(all_players)
        df.to_csv("players_data.csv", index=False)
        print("Successfully saved players_data.csv")

if __name__ == "__main__":
    fetch_squad_data()
