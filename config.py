
config = {
        'dk_salary_filename': 'Input/DKSalaries_MLB_20170411.csv',
        'sport': 'nba',  # 'nba', 'nfl', 'mlb', or 'debug'
        'scrape_draft_kings_salaries': True,
        'scrape_roto_grinders_nba': False,
        'random_initial_q_value': True, # True or False
        'reward':  {'AvgPointsPerGame': 1.0, 'Salary': 0.0},  # {metric: weight} for calculating reward
        'learning_rounds': 100  ,# number of rounds to iterate through lineup
        'log_qtable': True

    }
