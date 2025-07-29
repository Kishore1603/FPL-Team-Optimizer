from flask import Flask, jsonify, request, render_template
from src.data_fetcher import fetch_player_data, fetch_fixtures, fetch_team_data
from src.data_processor import prepare_dataset
from src.model_trainer import train_model, load_model
from src.team_optimizer import TeamOptimizer
import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

FPL_ID = 201605 # Use your FPL ID here (Have mentioned how to find this ID in the README.md)

def get_fpl_entry_info(fpl_id):
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def fetch_current_gw():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for event in data['events']:
            if event['is_current']:
                return event['id']
    return None

def fetch_team_picks(fpl_id, gw):
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/{gw}/picks/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def generate_fdr(opponent_team_id, teams):
    team = next((t for t in teams if t.get('id') == opponent_team_id), None)
    if team and 'previous_season_rank' in team:
        rank = team['previous_season_rank']
        if rank <= 4:
            return 5
        elif rank <= 10:
            return 4
        elif rank <= 15:
            return 3
        else:
            return 2
    return random.randint(2, 5)

def fixture_adjustment(fdr):
    return {2: 1.2, 3: 1.0, 4: 0.85, 5: 0.7}.get(fdr, 1.0)

def get_injury_severity(player):
    chance = player.get('chance_of_playing_next_round', 100)
    news = player.get('news', '').lower()
    import re
    date_match = re.search(r'(\d{1,2} [A-Za-z]{3})', news)
    if date_match:
        try:
            return_date = datetime.strptime(date_match.group(1) + f" {datetime.now().year}", "%d %b %Y")
            days_out = (return_date - datetime.now()).days
        except Exception:
            days_out = 0
    else:
        days_out = 0
    return (100 - (chance if chance is not None else 100)) + days_out

def normalize_position(pos):
    if pos in ['GK', 'GKP']:
        return 'GKP'
    if pos in ['DEF']:
        return 'DEF'
    if pos in ['MID']:
        return 'MID'
    if pos in ['FWD', 'FW']:
        return 'FWD'
    return pos

def map_team_names(players, teams):
    team_id_to_name = {int(t['id']): t['name'] for t in teams if 'id' in t and 'name' in t}
    for player in players:
        if 'team' in player:
            team_val = player['team']
            try:
                team_id = int(team_val)
            except Exception:
                team_id = team_val
            player['team_name'] = team_id_to_name.get(team_id, str(team_val))

@app.route('/', methods=['GET', 'POST'])
def index():
    fpl_info = recommended_team = total_points_used = current_team = transfer = captain = vice_captain = None
    total_expected_points = 0
    main_11 = []
    bench_4 = []
    if request.method == 'POST':
        # Data fetch and processing
        players = fetch_player_data()
        fixtures = fetch_fixtures()
        teams = fetch_team_data()

        # Prepare dataset and ensure player name is included
        processed_data = prepare_dataset(players, fixtures)
        # Build a mapping from player id to web_name
        id_to_name = {p['id']: p.get('web_name', '') for p in players if 'id' in p}
        for player in processed_data:
            if 'id' in player:
                player['web_name'] = id_to_name.get(player['id'], '')

        # Map team names for processed_data
        map_team_names(processed_data, teams)

        # Model
        model_filename = "model.xgb"
        try:
            model = load_model(model_filename)
        except Exception:
            df = pd.DataFrame(processed_data)
            if 'total_points' not in df.columns:
                return render_template('index.html', error='Missing total_points in data')
            df['target'] = df['total_points']
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in ['id', 'total_points', 'target', 'expected_points']]
            train_df = df[feature_cols + ['target']]
            model, _ = train_model(train_df)

        df = pd.DataFrame(processed_data)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['id', 'total_points', 'expected_points']]
        try:
            preds = model.predict(df[feature_cols])
            for i, player in enumerate(processed_data):
                player['expected_points'] = float(preds[i])
        except Exception:
            for player in processed_data:
                player['expected_points'] = player.get('total_points', 0)

        for player in processed_data:
            player['cost'] = player.get('price', 0)

        # Always select a team with total points between 90 and 99 (inclusive)
        import random
        max_players_per_position = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        optimized_team = []
        total_points_used = 0
        max_attempts = 30
        for _ in range(max_attempts):
            budget = round(random.uniform(90, 100), 2)
            optimizer = TeamOptimizer(processed_data, budget, max_players_per_position)
            team = optimizer.optimize_team()
            # Sort team for points calculation
            position_order = ['GKP', 'DEF', 'MID', 'FWD']
            grouped = {k: [] for k in position_order}
            for player in team:
                pos = normalize_position(player.get('position', ''))
                if pos in grouped:
                    grouped[pos].append(player)
                else:
                    grouped.setdefault(pos, []).append(player)
            sorted_team = []
            for pos in position_order:
                sorted_team.extend(grouped.get(pos, []))
            points = sum(float(p.get('price', p.get('cost', 0))) for p in sorted_team)
            if 90.0 <= points <= 100.0:
                optimized_team = team
                total_points_used = points
                break
        else:
            # Fallback: use last attempt even if not in range
            optimized_team = team
            total_points_used = points

        # Map team names for optimized_team
        map_team_names(optimized_team, teams)

        # Sort team for frontend: GK, DEF, MID, FWD
        position_order = ['GKP', 'DEF', 'MID', 'FWD']
        grouped = {k: [] for k in position_order}
        for player in optimized_team:
            pos = normalize_position(player.get('position', ''))
            if pos in grouped:
                grouped[pos].append(player)
            else:
                grouped.setdefault(pos, []).append(player)
        sorted_team = []
        for pos in position_order:
            sorted_team.extend(grouped.get(pos, []))

        # Ensure team_name is set for all players in sorted_team
        map_team_names(sorted_team, teams)
        for player in sorted_team:
            player['total_points'] = player.get('total_points', 0)
            player['expected_points'] = player.get('expected_points', 0)


        # Explicitly select main and bench for each position: GK(1+1), DEF(4+1), MID(4+1), FWD(2+1)
        def pick_best(players, n):
            return sorted(players, key=lambda p: p.get('expected_points', 0), reverse=True)[:n]

        # Always enforce 1 GK, 4 DEF, 4 MID, 2 FWD in the top 11
        main_gks = pick_best(grouped['GKP'], 1)
        main_defs = pick_best(grouped['DEF'], 4)
        main_mids = pick_best(grouped['MID'], 4)
        main_fwds = pick_best(grouped['FWD'], 2)
        starters = main_gks + main_defs + main_mids + main_fwds

        # Bench: next best in each position (excluding those already in starters)
        bench_gks = pick_best([p for p in grouped['GKP'] if p not in main_gks], 1)
        bench_defs = pick_best([p for p in grouped['DEF'] if p not in main_defs], 1)
        bench_mids = pick_best([p for p in grouped['MID'] if p not in main_mids], 1)
        bench_fwds = pick_best([p for p in grouped['FWD'] if p not in main_fwds], 1)
        bench = bench_gks + bench_defs + bench_mids + bench_fwds

        # Fill up to 15 with any remaining players (shouldn't be needed, but for safety)
        all_selected = starters + bench
        if len(all_selected) < 15:
            pool = [p for p in sorted_team if p not in all_selected]
            pool = sorted(pool, key=lambda p: p.get('expected_points', 0), reverse=True)
            all_selected += pool[:15-len(all_selected)]

        # Calculate total points/expected points for all 15 players, and for main 11 and 4 bench
        total_points_used = sum(float(p.get('price', p.get('cost', 0))) for p in sorted_team)
        total_expected_points = sum(float(p.get('expected_points', 0)) for p in sorted_team)

        # Explicitly select and expose main XI and bench 4 for the template
        main_11 = starters  # 1 GK, 4 DEF, 4 MID, 2 FWD
        bench_4 = []
        # Always try to fill bench 4 as 1 GK, 1 DEF, 1 MID, 1 FWD if possible
        if bench_gks:
            bench_4.append(bench_gks[0])
        if bench_defs:
            bench_4.append(bench_defs[0])
        if bench_mids:
            bench_4.append(bench_mids[0])
        if bench_fwds:
            bench_4.append(bench_fwds[0])
        # If less than 4, fill with next highest expected points from remaining bench
        if len(bench_4) < 4:
            already = set(p['id'] for p in bench_4)
            extra = [p for p in bench if p['id'] not in already]
            extra = sorted(extra, key=lambda p: p.get('expected_points', 0), reverse=True)
            bench_4 += extra[:4-len(bench_4)]

        main_11_expected = sum(float(p.get('expected_points', 0)) for p in main_11)
        bench_4_expected = sum(float(p.get('expected_points', 0)) for p in bench_4)

        recommended_team = sorted_team

        # FPL info
        entry_info = get_fpl_entry_info(FPL_ID)
        gw = fetch_current_gw()
        fpl_info = None
        if entry_info:
            fpl_info = {
                'player_name': f"{entry_info.get('player_first_name', '')} {entry_info.get('player_last_name', '')}",
                'team_name': entry_info.get('name', ''),
                'overall_points': entry_info.get('summary_overall_points', ''),
                'overall_rank': entry_info.get('summary_overall_rank', ''),
                'value': entry_info.get('last_deadline_value', ''),
                'bank': entry_info.get('last_deadline_bank', ''),
                'total_transfers': entry_info.get('summary_transfers', ''),
                'gameweek': gw
            }

        # Current team
        picks_data = fetch_team_picks(FPL_ID, gw) if gw else None
        current_team = []
        if picks_data and 'picks' in picks_data:
            player_id_map = {p['id']: p for p in processed_data if 'id' in p}
            for pick in picks_data['picks']:
                pid = pick['element']
                player = player_id_map.get(pid)
                if player:
                    current_team.append(player)

        # Logic for GW1 or no current team: show only recommended team and captaincy
        # Else: show only transfer and captaincy
        recommended_team = sorted_team
        transfer = None
        captain = vice_captain = None
        if (gw == 1) or (not current_team):
            # Set total_points to 0 for each recommended player in GW1 or no team
            for player in recommended_team:
                player['total_points'] = 0
            # GW1 or no team: show recommended team and captaincy
            def get_captains(team):
                sorted_team = sorted(team, key=lambda p: p.get('expected_points', 0), reverse=True)
                captain = sorted_team[0] if sorted_team else None
                vice_captain = sorted_team[1] if len(sorted_team) > 1 else None
                return captain, vice_captain
            captain, vice_captain = get_captains(sorted_team)
        else:
            # Later GWs: show only transfer and captaincy, enforce total points used <= 100
            best_out = None
            best_in = None
            best_gain = 0
            new_team = []
            transfer_reason = None
            injured_players = [p for p in current_team if p.get('chance_of_playing_next_round', 100) < 100 or ('injur' in p.get('news', '').lower() and p.get('news', ''))]
            current_ids = set(p['id'] for p in current_team)
            def team_count_check(temp_team):
                from collections import Counter
                team_ids = [p['team'] if isinstance(p['team'], int) else int(p['team']) for p in temp_team]
                counts = Counter(team_ids)
                return all(v <= 3 for v in counts.values())
            def team_total_points(team):
                position_order = ['GKP', 'DEF', 'MID', 'FWD']
                grouped = {k: [] for k in position_order}
                for player in team:
                    pos = normalize_position(player.get('position', ''))
                    if pos in grouped:
                        grouped[pos].append(player)
                    else:
                        grouped.setdefault(pos, []).append(player)
                sorted_team = []
                for pos in position_order:
                    sorted_team.extend(grouped.get(pos, []))
                return sum(float(p.get('price', p.get('cost', 0))) for p in sorted_team)

            if injured_players:
                injured_players.sort(key=get_injury_severity, reverse=True)
                out_player = injured_players[0]
                for in_player in optimized_team:
                    if in_player['id'] not in current_ids:
                        temp_team = [p for p in current_team if p['id'] != out_player['id']] + [in_player]
                        if not team_count_check(temp_team):
                            continue
                        temp_points = sum(p['expected_points'] for p in temp_team)
                        orig_points = sum(p['expected_points'] for p in current_team)
                        gain = temp_points - orig_points
                        total_pts = team_total_points(temp_team)
                        if gain > best_gain and total_pts <= 100:
                            best_gain = gain
                            best_out = out_player
                            best_in = in_player
            else:
                for out_player in current_team:
                    for in_player in optimized_team:
                        if in_player['id'] not in current_ids:
                            temp_team = [p for p in current_team if p['id'] != out_player['id']] + [in_player]
                            if not team_count_check(temp_team):
                                continue
                            temp_points = sum(p['expected_points'] for p in temp_team)
                            orig_points = sum(p['expected_points'] for p in current_team)
                            gain = temp_points - orig_points
                            total_pts = team_total_points(temp_team)
                            if gain > best_gain and total_pts <= 100:
                                best_gain = gain
                                best_out = out_player
                                best_in = in_player
            if best_in and best_out:
                new_team = [p for p in current_team if p['id'] != best_out['id']] + [best_in]
                transfer_reason = best_out.get('news', None)
                transfer = {
                    'out': best_out,
                    'in': best_in,
                    'reason': transfer_reason,
                    'gain': best_gain,
                    'new_team': new_team
                }
                def get_captains(team):
                    sorted_team = sorted(team, key=lambda p: p.get('expected_points', 0), reverse=True)
                    captain = sorted_team[0] if sorted_team else None
                    vice_captain = sorted_team[1] if len(sorted_team) > 1 else None
                    return captain, vice_captain
                captain, vice_captain = get_captains(new_team if new_team else current_team)

    return render_template('index.html',
        fpl_info=fpl_info,
        recommended_team=recommended_team,
        total_points_used=total_points_used,
        total_expected_points=total_expected_points,
        current_team=current_team,
        transfer=transfer,
        captain=captain,
        vice_captain=vice_captain,
        main_11=main_11,
        bench_4=bench_4
    )

if __name__ == '__main__':
    app.run(debug=True)