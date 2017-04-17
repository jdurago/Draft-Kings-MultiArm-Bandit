
config = {
        'dk_salary_filename': 'Input/DKSalaries.csv',
        'sport': 'nba',  # 'nba', 'nfl', 'mlb', 'pga' or 'debug'
        'scrape_draft_kings_salaries': True,
        'scrape_roto_grinders_nba': False,
        'random_initial_q_value': True, # True or False
        'reward':  {'AvgPointsPerGame': 1.0, 'Salary': 0.0},  # {metric: weight} for calculating reward
        'learning_rounds': 100  ,# number of rounds to iterate through lineup
        'log_qtable': True,
        # below are config for genetic algorithm
        'population_size': 500,
        'select_top_individuals': 20,
        'mutate_probability': 0.1,
        'generations': 500  # number of generations to evolve

    }
