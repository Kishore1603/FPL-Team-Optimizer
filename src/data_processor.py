def process_player_data(player_data):
    # Clean and transform player data
    cleaned_data = []
    # Map FPL element_type to position string
    position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    for player in player_data:
        cleaned_player = {
            'id': player['id'],
            'first_name': player['first_name'],
            'second_name': player['second_name'],
            'team': player['team'],
            'position': position_map.get(player.get('element_type'), 'UNK'),
            'total_points': player['total_points'],
            'form': player['form'],
            'price': player['now_cost'] / 10  # Convert price to float
        }
        cleaned_data.append(cleaned_player)
    
    return cleaned_data

def aggregate_fixture_data(fixture_data):
    # Aggregate fixture data for analysis
    aggregated_data = {}
    for fixture in fixture_data:
        home_team = fixture['team_h']
        away_team = fixture['team_a']
        if home_team not in aggregated_data:
            aggregated_data[home_team] = {'home_games': 0, 'home_goals': 0, 'away_games': 0, 'away_goals': 0}
        if away_team not in aggregated_data:
            aggregated_data[away_team] = {'home_games': 0, 'home_goals': 0, 'away_games': 0, 'away_goals': 0}
        
        aggregated_data[home_team]['home_games'] += 1
        aggregated_data[home_team]['home_goals'] += fixture.get('team_h_score') or 0
        aggregated_data[away_team]['away_games'] += 1
        aggregated_data[away_team]['away_goals'] += fixture.get('team_a_score') or 0
    
    return aggregated_data

def prepare_dataset(player_data, fixture_data):
    # Prepare the final dataset for model training
    processed_players = process_player_data(player_data)
    aggregated_fixtures = aggregate_fixture_data(fixture_data)
    
    # Combine player and fixture data into a single dataset
    dataset = []
    for player in processed_players:
        player_stats = {
            'id': player['id'],
            'name': f"{player['first_name']} {player['second_name']}",
            'team': player['team'],
            'position': player['position'],
            'total_points': player['total_points'],
            'form': player['form'],
            'price': player['price'],
            'home_games': aggregated_fixtures.get(player['team'], {}).get('home_games', 0),
            'home_goals': aggregated_fixtures.get(player['team'], {}).get('home_goals', 0),
            'away_games': aggregated_fixtures.get(player['team'], {}).get('away_games', 0),
            'away_goals': aggregated_fixtures.get(player['team'], {}).get('away_goals', 0)
        }
        dataset.append(player_stats)
    
    return dataset

# Added to support tests/test_data_processor.py
def clean_data(player_data):
    """
    Remove players with None in any of the key fields: first_name, second_name, team, points.
    """
    cleaned = []
    for player in player_data:
        if (
            player.get('first_name') is not None and
            player.get('second_name') is not None and
            player.get('team') is not None and
            player.get('points') is not None
        ):
            cleaned.append(player)
    return cleaned