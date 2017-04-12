"""
@author: joseph durago (jdurago@gmail.com)
@brief:
"""
# https://github.com/BenBrostoff/draft-kings-fun


## DO NOW
# TODO Implement reward system --> negative reward for going over cap limit, positive reward for improving point score
# TODO Implement q-learning for optimal draft team

## DO AFTER IMPLEMENTED
# TODO Implement advanced reward system --> if team is winning combo, update reward system
# TODO Save Q table and load

import sys
import random
import numpy as np
import pandas as pd
from config import config
from utils import import_draftkings_salaries, get_projections

import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='draft.log',level=logging.DEBUG, format='%(asctime)s %(message)s')


# sys.path.append('draft-kings-fun/scrapers')
# from fantasy_pros import scrape
# scrape()


class Create_Draft():

    def __init__(self, salaries, lineup):
        self.salaries = salaries # DK salaries dataframe
        self.lineup = lineup # list of positions needed in lineup ex. ['PG', 'PF', 'C'] or ['QB', 'WR', 'WR']
        self.best_lineup = {}

        # TODO initialize Q learning variables here
        self.q_table = self.salaries[['Name', 'Position']].copy()
        for position in lineup:
            self.q_table[str(position)] = 1.0 # Initialize Qscore table in self.salaries, will be used to determine actions. 1 is used to promote adding new players

        logging.debug('q_table')
        logging.debug(self.q_table)
        self.epsilon = 0.3  # variable for performing a random action. 1.0 = always does random action
        self.n_trials = 3  # number of times to go through lineup

    def initialize_lineup(self):
        """
        creates lineup by randomly selecting players

        :return: {'PG': 'Maurice Ndour', 'C': Tyus Jones, 'PF': Jordan Hill} #player is a row from salaries dataframe
        """
        df = self.salaries
        for position in self.lineup:
            try:
                if position != 'Util':
                    self.best_lineup.update({position: df[df['Position'].str.contains(position)].sample(1)})
                else:
                    self.best_lineup.update({'Util': df.sample(1)})
            except ValueError:
                print "Position Does Not Exist in Dataframe"
                raise

        return self.best_lineup

    def calculate_lineup_score(self, lineup):
        """
        :input
        lineup = {'PG': 'Maurice Ndour', 'C': Tyus Jones, 'PF': Jordan Hill} #player is a row from salaries dataframe

        :return: pd.DataFrame(
            Position_Total                                        SG/SFPF/CPFPGPGPFSGSF
            Name_Total                Bojan BogdanovicDirk NowitzkiLucas NogueiraPat...
            Salary_Total                                                          34700
            GameInfo_Total            Was@Cle 07:30PM ETTor@Dal 08:30PM ETTor@Dal 08...
            AvgPointsPerGame_Total                                               144.95
            teamAbbrev_Total                                     WasDalTorSASACleDalMin
            DollarPerPoint_Total                                                2917.09
            dtype: object
        )
        """
        df = pd.DataFrame()

        for position, player in self.best_lineup.iteritems():
            df = df.append(player)

        column_names = df.columns
        column_names = [name + '_Total' for name in column_names]
        df.columns = column_names
        return df.sum()

    def reset(self):
        # TODO update Q learning variables here

        # self.alpha =
        pass

    def add_player(self, position, player):
        """
        add player to self.best_lineup

        :return: float(reward) for adding player
        """
        pass

    def update(self, trial_num):
        old_lineup = self.best_lineup
        initial_score = self.calculate_lineup_score(old_lineup)
        logging.info('trial_num: '+ str(trial_num) + ' initial_score: ' + str(initial_score))
        actionList = ['AddPlayer', 'DontAddPlayer']

        # Updates the QTable as we iterate through all the players
        for index, new_player in self.salaries.iterrows():
            logging.debug('update Q table: ', str(index), str(new_player))
            # Select action with maximum Q value if less than epsilon, otherwise choose random action
            # If Q value is greater than 0, it will add the player. Otherwise we don't add the player
            randNumber = np.random.uniform(low=0.0, high=1.0)

            current_player = self.q_table.query(""" Name == "{}" """.format(new_player.Name)).max(numeric_only=True)
            if randNumber > self.epsilon:

                player_position = current_player.idxmax()
                player_qscore = current_player.max()

                logging.debug('player')
                logging.debug(current_player)
                logging.debug('player_best_position: ')
                logging.debug(player_position)
                logging.debug('player_best_qscore: ')
                logging.debug(player_qscore)
                # Additional limitations for actions

                # Don't add negative q score players
                # if player_best_qscore > 0:
                #     action = 'AddPlayer'
                # else:
                #     action = 'DontAddPlayer'
            else:
                player_position = current_player

                # Execute Action and Get Reward
            # TODO Add negative points for adding player to incorrect position ex. adding PG to C position


            # Update Q Table based on the following eq: Q(s,a) < - Q(s,a) + alpha[reward + gamma(Q(s', a')) - Q(s,a)]


        return trial_num

    def optimize_lineup(self):

        for trial_num in range(self.n_trials):
            print 'TRIAL - ', trial_num
            self.update(trial_num)

if __name__ == '__main__':
    logging.info('_____________________________')
    logging.info('Script Start')
    # dk_salaries = import_draftkings_salaries('Input/DKSalaries.csv')
    # dk_salaries.to_pickle('Input/DKSalaries.pkl')

    dk_salaries = get_projections('Input/DKSalaries.pkl')
    #print dk_salaries.Position.unique()

    if config['sport'] == 'nba':
        lineup = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'Util']
    elif config['sport'] == 'debug':
        lineup = ['PG', 'SG', 'SF']
    else:
        raise ValueError('incorrect value in config')

    my_draft = Create_Draft(salaries=dk_salaries, lineup=lineup)
    initial_lineup = my_draft.initialize_lineup()
    logging.info('Lineup: ' + str(initial_lineup))

    my_lineup = my_draft.optimize_lineup()

    print my_lineup

    logging.info('Script Complete')



