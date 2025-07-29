from scipy.optimize import linprog
import numpy as np

class TeamOptimizer:
    def get_team_structure(self):
        # Count number of players per position in the optimized team
        team = self.optimize_team()
        structure = {'goalkeepers': 0, 'defenders': 0, 'midfielders': 0, 'forwards': 0}
        for player in team:
            if player['position'] == 'GKP':
                structure['goalkeepers'] += 1
            elif player['position'] == 'DEF':
                structure['defenders'] += 1
            elif player['position'] == 'MID':
                structure['midfielders'] += 1
            elif player['position'] == 'FWD':
                structure['forwards'] += 1
        return structure

    def select_players_based_on_performance(self):
        # Select all players with points > 0
        return [player for player in self.player_data if player.get('points', 0) > 0]
    def __init__(self, player_data, budget, max_players_per_position):
        self.player_data = player_data
        self.budget = budget
        # Accept dict for max_players_per_position, fallback to int for backward compatibility
        if isinstance(max_players_per_position, dict):
            self.max_players_per_position = max_players_per_position
        else:
            self.max_players_per_position = {'GKP': max_players_per_position, 'DEF': max_players_per_position, 'MID': max_players_per_position, 'FWD': max_players_per_position}

    def optimize_team(self):
        num_players = len(self.player_data)
        costs = np.array([player['cost'] for player in self.player_data])
        points = np.array([player['expected_points'] for player in self.player_data])
        
        # Objective function: maximize points
        c = -points
        
        # Constraints
        # Budget constraint as inequality (sum(costs * x) <= budget)
        A_ub = [costs]
        b_ub = [self.budget]

        # Position constraints as equalities
        A_eq = []
        b_eq = []
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            position_mask = np.array([player['position'] == position for player in self.player_data])
            A_eq.append(position_mask)
            b_eq.append(self.max_players_per_position.get(position, 0))

        # Bounds for each player (0 or 1)
        bounds = [(0, 1) for _ in range(num_players)]

        # Solve the linear programming problem
        result = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method='highs'
        )

        if result.success:
            selected_players = np.round(result.x).astype(int)
            return [self.player_data[i] for i in range(num_players) if selected_players[i] == 1]
        else:
            raise ValueError("Optimization failed: " + result.message)