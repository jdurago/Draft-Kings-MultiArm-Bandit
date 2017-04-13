from config import config
from utils import import_draftkings_salaries
import numpy as np
import random
import itertools

import logging
# create logger with __name__
logger = logging.getLogger('my_log')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('draft2.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# DK_SALARIES_FILE = 'Input/DKSalaries_Debug.csv'
DK_SALARIES_FILE = config['dk_salary_filename']

# TODO update q table so that players only have states with eligible positions ie. Lebron James(SF) can't be a PG, this will speed up processing time

class CreateDraft():
    def __init__(self, dk_data, dk_lineup):
        """
        optimized_lineup = {
            'PG': 'Kobe Bryant'
            'SG': 'DeMar DeRozan'
        }

        """
        # self.action_list = ['add_player', 'dont_add_player']
        self.action_list = ['add_player']
        self.dk_data = dk_data  # pd.DataFrame of draft kings salary
        self.dk_lineup = dk_lineup  # desired lineup to input into draft kings, ['PG', 'SG', 'Util']
        # stateList = ['PG|add_player', 'PG|dont_add_player', 'C|add_player']
        self.stateList = [item[0] + '|' + item[1] for item in itertools.product(self.dk_lineup, self.action_list )]
        self.current_round = 0  # round for iterating through lineup for q-learning

        # Equation for updating q values: Q(s,a) < - Q(s,a) + alpha[reward + gamma(Q(s', a')) - Q(s,a)]
        self.initial_q_value = 1.0
        self.epsilon = 0.3  # variable for randomness, 0.0 = no random actions taken; 1.0 = random action always taken
        self.alpha = 0.5  # learning rate, ie to what extent the newly acquired information will override the old information, alpha = 0.0 agent doesn't learn anything, alpha = 1.0 agent only considers most recent info
        self.gamma = 0.00  # discount factor, determines importance of future rewards, gamma = 0 only considers current rewards, gamma = 1.0 will strive for long term rewards

        self.previousPlayer = self.dk_data['Name'].sample(1).values[0]
        self.previousAction = self.stateList[0]
        self.previousReward = 0.0

        self.optimized_lineup = self.initialize_lineup()
        self.qtable = self.initialize_qtable()

    def initialize_lineup(self):
        """
        creates lineup by randomly selecting players

        :return: {'PG': 'Maurice Ndour', 'C': Tyus Jones, 'PF': Jordan Hill} #player is a row from salaries dataframe
        """
        df = self.dk_data
        lineup = {}
        for position in self.dk_lineup:
            try:
                if position != 'Util':
                    lineup.update({position: df['Name'].sample(1).values[0]})
                else:
                    lineup.update({'Util': df['Name'].sample(1).values[0]})
            except ValueError:
                print "Position Does Not Exist in Dataframe"
                raise

        logger.info('Initial Lineup')
        logger.info(lineup)

        return lineup

    def initialize_qtable(self):
        """
        qtable data structure: {'player_name': {'Position'|'Add Or Dont Add'}}

        qtable: {'Damian Lillard': {'PG|dont_add_player': 1.0, 'PG|add_player': 1.0, 'C|dont_add_player': 1.0, 'C|add_player': 1.0},
                'Lebron James':  {'PG|dont_add_player': 1.0, 'PG|add_player': 1.0, 'C|dont_add_player': 1.0, 'C|add_player': 1.0}


        """
        qtable = {}

        for name in self.dk_data['Name'].values:
            if config['random_initial_q_value'] is True:
                qtable[name] = {state: np.random.uniform(low=0.01, high=0.1) for state in self.stateList}
            else:
                qtable[name] = {state: self.initial_q_value for state in self.stateList}

        if config['log_qtable'] is True:
            logger.debug('Initial QTable')
            logger.debug(qtable)

        return qtable

    def calculate_lineup_score(self, lineup):
        """
        :param lineup: {'C': 'Kawhi Leonard', 'SG': 'DeMar DeRozan', 'PG': 'LeBron James'}
        :return: float 123.32

        # lineup_score = sum(salary) + avg(points_per_game)
        """

        data = self.dk_data
        filtered_lineup = data[data['Name'].isin(lineup.values())]

        lineup_score = 0
        # adds all columns in  config['rewards'] to calculate the lineup score
        for metric, weight in config['reward'].iteritems():
            lineup_score += filtered_lineup.sum()[metric] * weight

        # returns summed lineup_score, total salary cap, and expected total points
        return lineup_score, filtered_lineup.sum()['Salary'], filtered_lineup.sum()['AvgPointsPerGame']

    def calculate_reward(self, old_lineup, new_lineup, player_name):

        old_line_score, _, _ = self.calculate_lineup_score(old_lineup)
        new_lineup_score, new_lineup_salary, _ = self.calculate_lineup_score(new_lineup)

        reward = new_lineup_score - old_line_score

        # Adds negative-reward if salary cap goes above $50k
        if new_lineup_salary > 50000.0:
            reward += -1 * (new_lineup_salary - 50000.0)
        elif new_lineup_salary > 45000.0 and new_lineup_salary <= 50000.0:
            reward += 50000.0 - new_lineup_salary
        elif new_lineup_salary < 40000:
            reward += -1000.0

        # Adds negative-reward for adding player in incorrect position
        dk_player_position = self.dk_data.query('Name == "{}"'.format(player_name)).Position.values[0]  # allocated position in draft kings
        for position, player in new_lineup.iteritems():
            if player == player_name:

                if position != 'Util':
                    if position not in dk_player_position:
                        reward += -100000


        return reward

    def execute_action(self, player_name, action, lineup):
        """
        calculates difference in reward when performing action, however the overall lineup is not modified
        """
        old_lineup = lineup.copy()
        new_lineup = lineup.copy()

        position = action.split('|')[0]
        if '|add_player' in action:
            new_lineup[position] = player_name
            # if player_name not in new_lineup.values():
            #     new_lineup[position] = player_name
            # else:
            #     logger.debug('Duplicate player found, but not added - {}'.format(player_name))

        elif '|dont_add_player' in action:
            logger.debug('{} - {}'.format(player_name, action))

        else:
            raise ValueError('Incorrect Action')

        reward = self.calculate_reward(old_lineup=old_lineup, new_lineup=new_lineup, player_name=player_name)

        return reward

    def update_qtable(self):

        # Updates the QTable as we iterate through all the players
        # new_player is pd.Series object, ex. new_player['Name'] == 'Lebron James'

        logger.debug('Updating QTable')
        temporary_lineup = self.optimized_lineup.copy()
        for index, new_player in self.dk_data.iterrows():

            randNumber = np.random.uniform(low=0.0, high=1.0)

            # Select action with maximum Q value if less than epsilon, otherwise choose random action
            if randNumber > self.epsilon:  # do normal routine
                action = max(self.qtable[new_player['Name']], key=self.qtable[new_player['Name']].get)
            else: # do random routine
                logger.debug('Did Something Random')
                action = random.choice(self.stateList)

            # Execute action and get reward
            reward = self.execute_action(new_player['Name'], action, temporary_lineup)

            logger.debug("round:{:<5} - player:{:<20} - action:{:<20} - randNumber:{:<20} - reward:{:<20}".format(self.current_round, new_player['Name'], action, randNumber, reward))

            # Update Q Table based on the following eq: Q(s,a) < - Q(s,a) + alpha[reward + gamma(Q(s', a')) - Q(s,a)]
            self.qtable[self.previousPlayer][self.previousAction] = self.qtable[self.previousPlayer][self.previousAction] + self.alpha * (self.previousReward + self.gamma * (self.qtable[new_player['Name']][action] - self.qtable[self.previousPlayer][self.previousAction] ))

            if config['log_qtable'] is True:
                logger.debug('qtable: {}'.format(self.qtable))

            # Update States
            self.previousPlayer = new_player['Name']
            self.previousAction = action
            self.previousReward = reward

    def check_player_in_temp_lineup(self, player_name, temp_lineup):
        """
        player_name = 'Seth Curry'
        temp_lineup = {'C': ('Seth Curry', 123),
                         'PG': ('Justin Holiday', 456),
                         'SG': ('Festus Ezeli', 456),
                         'Util': ('Wesley Matthews', 456)}
        """
        for position, (player, qscore) in temp_lineup.iteritems():
            if player_name == player:
                return True
        return False

    def optimize_lineup(self):
        # TODO after each loop, update self.alpha
        for current_round in range(config['learning_rounds']):
            self.current_round = current_round
            self.update_qtable()

            #  uses updated qtable and finds optimum lineup based on max value of q table for each position
            # TODO breakup below into separate function: def find_best_lineup(self):
            temp_lineup = {}
            for position in self.dk_lineup:  # ex. ['PG', 'C', 'SG', 'Util']
                temp_lineup.update({position: (None, None)})  # initializes temp_lineup with position and empty player and q value
                for player, state_dict in self.qtable.iteritems():  # ex. 'Marshall Plumlee': {'PG|dont_add_player': 0.0, 'Util|dont_add_player': 0.1}

                    if self.check_player_in_temp_lineup(player, temp_lineup):
                        logger.debug('Player Already In Lineup. Player: {} - Lineup: {}'.format(player, temp_lineup.values()))
                    else:
                        for state, qvalue in state_dict.iteritems():
                                if position == 'Util' and qvalue > temp_lineup[position][1] and '|add_player' in state:
                                    temp_lineup.update({position: (player, qvalue)})
                                elif position in state and qvalue > temp_lineup[position][1] and '|add_player' in state:
                                    temp_lineup.update({position: (player, qvalue)})

            optimized_lineup = {}
            for position, player in temp_lineup.iteritems():
                optimized_lineup.update({position: player[0]})

            self.optimized_lineup = optimized_lineup

            print 'round: {} - lineup:{} - score:{}'.format(self.current_round , self.optimized_lineup, self.calculate_lineup_score(self.optimized_lineup))
            logger.info('round: {} - lineup:{} - score:{}'.format(self.current_round , self.optimized_lineup, self.calculate_lineup_score(self.optimized_lineup)))

        logger.info('Final Q Table: {}'.format(self.qtable))
        return self.optimized_lineup, self.calculate_lineup_score(self.optimized_lineup)

if __name__ == '__main__':
    logger.info('___________')
    logger.info('Script Start')

    dk_data = import_draftkings_salaries(DK_SALARIES_FILE)

    if config['sport'] == 'debug':
        desired_lineup = ['PG', 'C', 'SG', 'Util']
    elif config['sport'] == 'nba':
        desired_lineup = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'Util']
    elif config['sport'] == 'mlb':
        desired_lineup = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']

    logger.info('DK Salaries: ')
    logger.info(dk_data)

    my_draft = CreateDraft(dk_data, desired_lineup)
    my_lineup, final_score = my_draft.optimize_lineup()
    print 'Final Lineup and Expected Score'
    print my_lineup, final_score
    logger.info('Final Lineup: {} - Final Score: {}'.format(my_lineup, final_score))

    logger.info('Script Complete')
    logger.info('___________')