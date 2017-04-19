from unittest import TestCase
import logging
import os
import collections

from genetic_algorithm import Individual, Population, check_player_in_lineup
from utils import import_draftkings_salaries
from config import config

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'DKSalaries_Debug.csv')


class TestIndividual(TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

        self.old_config = config
        config['reward'] = {'AvgPointsPerGame': 0.5, 'Salary': 0.5}
        config['population_size'] = 10
        config['select_top_individuals'] = 2
        config['mutate_probability'] = 0.1
        config['generations'] = 2

        self.dk_data = import_draftkings_salaries(TESTDATA_FILENAME)
        self.desired_lineup = [('PG', 'PG'), ('C', 'C'), ('SG', 'SG'), ('Util', 'Util')]

    def test_check_player_in_lineup_True(self):

        lineup = {'C': 'LeBron James', 'G': 'John Wall', 'SG': 'DeMar DeRozan'}
        player = 'LeBron James'
        my_individual = Individual(self.dk_data, self.desired_lineup)

        self.assertEquals(check_player_in_lineup(lineup, player), True)

    def test_check_player_in_lineup_False(self):

        lineup = {'C': 'LeBron James', 'G': 'John Wall', 'SG': 'DeMar DeRozan'}
        player = 'Damian Lillard'
        my_individual = Individual(self.dk_data, self.desired_lineup)

        self.assertEquals(check_player_in_lineup(lineup, player), False)

    def test_initialize_lineup(self):
        my_individual = Individual(self.dk_data, self.desired_lineup)
        my_lineup = my_individual.initialize_lineup()

        self.assertEquals(len(my_lineup), 4)

    def test_get_fitness(self):
        my_lineup = {'C': 'Karl-Anthony Towns', 'PG': 'John Wall', 'SG': 'DeMar DeRozan'}
        my_individual = Individual(self.dk_data, self.desired_lineup)
        my_fitness = my_individual.get_fitness(my_lineup)

        self.assertEquals(my_fitness, 14619.183)

    def test_get_fitness_above_salary_cap(self):
        my_lineup = {'C': 'Karl-Anthony Towns', 'PG': 'Kawhi Leonard', 'SG': 'LeBron James', 'A': 'John Wall', 'B': 'Damian Lillard', 'C': 'DeMar DeRozan', 'D': 'Kyrie Irving'}
        my_individual = Individual(self.dk_data, self.desired_lineup)
        my_fitness = my_individual.get_fitness(my_lineup)

        self.assertEquals(my_fitness, -1)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        config = self.old_config


class TestPopulation(TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.old_config = config

        self.old_config = config
        config['reward'] = {'AvgPointsPerGame': 0.5, 'Salary': 0.5}

        self.dk_data = import_draftkings_salaries(TESTDATA_FILENAME)
        self.desired_lineup = [('PG', 'PG'), ('C', 'C'), ('SG', 'SG'), ('Util', 'Util')]

    def test_initialize_population(self):
        # TODO assert actual value of lineup instead of type
        pop_size = config['population_size']

        my_population = Population(self.dk_data, self.desired_lineup)
        my_individual = my_population.current_generation[pop_size - 1]
        self.assertEquals(type(my_individual.lineup), collections.OrderedDict)

    def test_selection(self):
        # TODO assert actual value of lineup instead of type

        my_population = Population(self.dk_data, self.desired_lineup)
        # verify if my_population.selection() properly selects top individuals in current_generation
        for individual in my_population.current_generation:
            print individual.fitness, individual.lineup

        # self.assertEquals(len(my_population.selection()), config['select_top_individuals'])
        self.assertEquals(type(my_population.selection()), list)

    def test_crossover_old(self):

        # Initialize parent 1 and set known lineup
        parent1 = Individual(self.dk_data, self.desired_lineup)
        d = collections.OrderedDict()
        d['PG'] = 'Damian Lillard'
        d['C'] = 'Dewayne Dedmon'
        d['SG'] = 'DeMar DeRozan'
        d['Util'] = 'Kawhi Leonard'
        parent1.lineup = d

        # Initialize parent 2 and set known lineup
        parent2 = Individual(self.dk_data, self.desired_lineup)
        d = collections.OrderedDict()
        d['PG'] = 'John Wall'
        d['C'] = 'Karl-Anthony Towns'
        d['SG'] = 'James Jones'
        d['Util'] = 'Kyrie Irving'
        parent2.lineup = d

        my_population = Population(self.dk_data, self.desired_lineup)
        crossover_point = 2

        child1, child2 = my_population.crossover_old(parent1, parent2, crossover_point)

        assert_child1 = collections.OrderedDict([('PG', 'Damian Lillard'), ('C', 'Dewayne Dedmon'), ('SG', 'James Jones'), ('Util', 'Kyrie Irving')])
        self.assertEquals(child1.lineup, assert_child1)

        assert_child2 = collections.OrderedDict([('PG', 'John Wall'), ('C', 'Karl-Anthony Towns'), ('SG', 'DeMar DeRozan'), ('Util', 'Kawhi Leonard')])
        self.assertEquals(child2.lineup, assert_child2)

    def test_crossover(self):

        # Initialize parent 1 and set known lineup
        parent1 = Individual(self.dk_data, self.desired_lineup)
        d = collections.OrderedDict()
        d['PG'] = 'Damian Lillard'
        d['C'] = 'Dewayne Dedmon'
        d['SG'] = 'DeMar DeRozan'
        d['Util'] = 'Kawhi Leonard'
        parent1.lineup = d

        # Initialize parent 2 and set known lineup
        parent2 = Individual(self.dk_data, self.desired_lineup)
        d = collections.OrderedDict()
        d['PG'] = 'John Wall'
        d['C'] = 'Karl-Anthony Towns'
        d['SG'] = 'James Jones'
        d['Util'] = 'Kyrie Irving'
        parent2.lineup = d

        my_population = Population(self.dk_data, self.desired_lineup)
        crossover_point = 2

        children = my_population.crossover(parent1, parent2)

        assert_child1 = collections.OrderedDict([('PG', 'Damian Lillard'), ('C', 'Dewayne Dedmon'), ('SG', 'DeMar DeRozan'), ('Util', 'Kawhi Leonard')])
        self.assertEquals(children[0].lineup, assert_child1)

    def test_mutation(self):
        # TODO assert actual value of lineup instead of type

        my_population = Population(self.dk_data, self.desired_lineup)
        child = Individual(self.dk_data, self.desired_lineup)
        child.lineup = collections.OrderedDict([('PG', 'Damian Lillard'), ('C', 'Dewayne Dedmon'), ('SG', 'James Jones'), ('Util', 'Kyrie Irving')])
        mutated_child = my_population.mutation(child, rand_value=0.001)
        self.assertEquals(type(mutated_child.lineup),collections.OrderedDict)

    def test_next_generation(self):

        my_population = Population(self.dk_data, self.desired_lineup)
        my_population.next_generation()

        self.assertEquals(type(my_population.next_generation()), list)

    def test_evolve(self):

        my_population = Population(self.dk_data, self.desired_lineup)
        top_indiviual = my_population.evolve()
        self.assertEquals(type(top_indiviual.lineup), collections.OrderedDict)


    def tearDown(self):
        logging.disable(logging.NOTSET)
        config = self.old_config