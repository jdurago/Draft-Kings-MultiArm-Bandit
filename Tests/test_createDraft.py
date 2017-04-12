from unittest import TestCase
from create_draft2 import CreateDraft
from utils import import_draftkings_salaries
import logging
from config import config
import os

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'DKSalaries_Debug.csv')


class TestCreateDraft(TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.dk_data = import_draftkings_salaries(TESTDATA_FILENAME)
        self.my_draft = CreateDraft(self.dk_data, dk_lineup=['PG', 'C', 'SG'])

        self.old_config = config
        config['reward'] = {'AvgPointsPerGame': 0.5, 'Salary': 0.5}

    def test_initialize_lineup(self):
        # verify initialized lineup is a dict
        # TODO assert if subset is part of initalized lineup
        print self.my_draft.initialize_lineup()
        self.assertEquals(type(self.my_draft.initialize_lineup()), dict)

    def test_initialize_qtable(self):
        # verify a value within q table is initialized to initial_q_value variable
        my_dict = self.my_draft.initialize_qtable()

        if config['random_initial_q_value'] is True:
            self.assertEquals(type(my_dict), dict)
        else:
            self.assertDictContainsSubset({'Damian Lillard': {'PG|dont_add_player': 1.0, 'SG|add_player': 1.0,
                                                              'C|add_player': 1.0, 'SG|dont_add_player': 1.0,
                                                              'C|dont_add_player': 1.0, 'PG|add_player': 1.0}}, my_dict)

    def test_execute_action(self):
        player_name = 'LeBron James'
        action = 'PG|dont_add_player'

        value = self.my_draft.execute_action(player_name, action)
        self.assertEquals(value, 0.0)

    def test_calculate_lineup_score(self):
        lineup = {'C': 'Kawhi Leonard', 'SG': 'DeMar DeRozan', 'PG': 'LeBron James'}
        score, _, _ = self.my_draft.calculate_lineup_score(lineup)

        self.assertEquals(score, 14618.706)

    def test_calculate_reward(self):

        old_lineup = {'C': 'Kawhi Leonard', 'SG': 'DeMar DeRozan', 'PG': 'LeBron James'}
        new_lineup = {'C': 'Kawhi Leonard', 'SG': 'DeMar DeRozan', 'PG': 'Kyrie Irving'}

        reward = self.my_draft.calculate_reward(old_lineup, new_lineup)
        self.assertEquals(reward, -1255.6445000000003)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        config = self.old_config



