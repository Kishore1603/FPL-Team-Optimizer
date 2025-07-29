import requests

def fetch_player_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    return data['elements']

def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()
        except Exception as e:
            print(f"Error decoding JSON from fixtures API: {e}")
            return []
    else:
        print(f"Failed to fetch fixtures. Status: {response.status_code}")
        return []

def fetch_team_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    return data['teams']