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
fh.setLevel(logging.DEBUG)
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
DK_SALARIES_FILE = 'Input/DKSalaries.csv'


class CreateDraft():
    def __init__(self, dk_data, dk_lineup):
        """
        optimized_lineup = {
            'PG': 'Kobe Bryant'
            'SG': 'DeMar DeRozan'
        }

        """
        self.action_list = ['add_player', 'dont_add_player']
        self.dk_data = dk_data  # pd.DataFrame of draft kings salary
        self.dk_lineup = dk_lineup  # desired lineup to input into draft kings, ['PG', 'SG', 'Util']
        # stateList = ['PG|add_player', 'PG|dont_add_player', 'C|add_player']
        self.stateList = [item[0] + '|' + item[1] for item in itertools.product(self.dk_lineup, self.action_list )]
        self.current_round = 0  # round for iterating through lineup for q-learning

        # Equation for updating q values: Q(s,a) < - Q(s,a) + alpha[reward + gamma(Q(s', a')) - Q(s,a)]
        self.initial_q_value = 1.0
        self.epsilon = 0.3  # variable for randomness, 0.0 = no random actions taken; 1.0 = random action always taken
        self.alpha = 1.0  # learning rate, ie to what extent the newly acquired information will override the old information, alpha = 0.0 agent doesn't learn anything, alpha = 1.0 agent only considers most recent info
        self.gamma = 0.0  # discount factor, determines importance of future rewards, gamma = 0 only considers current rewards, gamma = 1.0 will strive for long term rewards

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
                    lineup.update({position: df[df['Position'].str.contains('SF')]['Name'].sample(1).values[0]})
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

        logger.debug('QTable')
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

    def calculate_reward(self, old_lineup, new_lineup):

        old_line_score, _, _ = self.calculate_lineup_score(old_lineup)
        new_lineup_score, _, _ = self.calculate_lineup_score(new_lineup)

        reward = new_lineup_score - old_line_score

        # TODO add negative-reward if salary goes above $50k
        # TODO add negative-reward for adding player in incorrect position
        return reward

    def execute_action(self, player_name, action):
        """
        updates self.optimized_lineup based on action and player
        """
        old_lineup = self.optimized_lineup.copy()

        logger.debug('Start Executing Action')

        position = action.split('|')[0]
        if '|add_player' in action:
            logger.debug('{} - {}'.format(player_name, action))

            if player_name not in self.optimized_lineup.values():
                self.optimized_lineup[position] = player_name
            else:
                logger.debug('Duplicate player found, but not added - {}'.format(player_name))

        elif '|dont_add_player' in action:
            logger.debug('{} - {}'.format(player_name, action))

        else:
            raise ValueError('Incorrect Action')

        logger.debug('Done Executing Action')

        reward = self.calculate_reward(old_lineup=old_lineup, new_lineup=self.optimized_lineup)
        return reward

    def update_qtable(self):

        # Updates the QTable as we iterate through all the players
        # new_player is pd.Series object, ex. new_player['Name'] == 'Lebron James'

        # TODO options for updating q table
        # TODO option 1: as we iterate through DKSalaries, search through entire qtable and add only players with best score in q table, pros: ; cons: new players less likely to be added
        # TODO option 2: have temporary qtable where we update all the players, then after looping through entire draft select players that maximize q table
        # TODO option 3:
        logger.info('Updating QTable')
        for index, new_player in self.dk_data.iterrows():

            randNumber = np.random.uniform(low=0.0, high=1.0)

            # Select action with maximum Q value if less than epsilon, otherwise choose random action
            if randNumber > self.epsilon:  # do normal routine
                # TODO search qtable for all players with highest q value for a position, instead of iterating through lineup
                # TODO Do above, current procedure adds last set of people in DKSalaries.csv
                action = max(self.qtable[new_player['Name']], key=self.qtable[new_player['Name']].get)
            else: # do random routine
                logger.debug('Did Something Random')
                action = random.choice(self.stateList)

            # Execute action and get reward
            # TODO add if statement; if player in self.optimized_lineup skip and don't update q table
            reward = self.execute_action(new_player['Name'], action)

            logger.debug("round:{:<5} - player:{:<20} - action:{:<20} - randNumber:{:<20} - reward:{:<20}".format(self.current_round, new_player['Name'], action, randNumber, reward))

            # Update Q Table based on the following eq: Q(s,a) < - Q(s,a) + alpha[reward + gamma(Q(s', a')) - Q(s,a)]
            self.qtable[self.previousPlayer][self.previousAction] = self.qtable[self.previousPlayer][self.previousAction] + self.alpha * (reward + self.gamma * (self.qtable[new_player['Name']][action] - self.qtable[self.previousPlayer][self.previousAction] ))

            logger.debug('qtable: {}'.format(self.qtable))

            # Update States
            self.previousPlayer = new_player['Name']
            self.previousAction = action
            self.previousReward = reward

    def optimize_lineup(self):
        # TODO after each loop, update self.alpha
        for current_round in range(config['learning_rounds']):
            self.current_round = current_round
            self.update_qtable()
            print self.optimized_lineup, self.calculate_lineup_score(self.optimized_lineup)
            logger.info('lineup:{} - score:{}'.format(self.optimized_lineup, self.calculate_lineup_score(self.optimized_lineup)))

        return self.optimized_lineup, self.calculate_lineup_score(self.optimized_lineup)

if __name__ == '__main__':
    logger.info('___________')
    logger.info('Script Start')

    dk_data = import_draftkings_salaries(DK_SALARIES_FILE)

    if config['sport'] == 'debug':
        desired_lineup = ['PG', 'C', 'SG', 'Util']
    elif config['sport'] == 'nba':
        desired_lineup = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'Util']

    logger.info('DK Salaries: ')
    logger.info(dk_data)

    my_draft = CreateDraft(dk_data, desired_lineup)
    my_lineup, final_score = my_draft.optimize_lineup()
    print 'Final Lineup and Expected Score'
    print my_lineup, final_score
    logger.info('Final Lineup: {} - Final Score: {}'.format(my_lineup, final_score))

    logger.info('Script Complete')
    logger.info('___________')