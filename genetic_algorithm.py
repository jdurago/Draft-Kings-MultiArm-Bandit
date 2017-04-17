from utils import import_draftkings_salaries
from config import config
import collections
import copy
import numpy as np
import random

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


DK_SALARIES_FILE = config['dk_salary_filename']


def check_player_in_lineup(lineup, player):
    if player in lineup.values():
        return True
    else:
        return False


class Individual:

    def __init__(self, dk_data, desired_lineup):
        self.dk_data = dk_data
        self.desired_lineup = desired_lineup

        self.lineup = self.initialize_lineup()
        self.fitness = self.get_fitness(self.lineup)

    def initialize_lineup(self):
        """
        creates random lineup
        verifies no duplicate players in lineup, if a player is already in lineup will keep randomly selecting player
        until a unique player is found
        """
        df = self.dk_data

        lineup = collections.OrderedDict()
        for position in self.desired_lineup:
            # Util position can be any player
            if position == 'Util':
                random_player = df.sample(1).Name.values[0]
                while check_player_in_lineup(lineup, random_player):
                    random_player = df.sample(1).Name.values[0]
                lineup.update({position: random_player})
            else:
                random_player = df[df['Position'].str.contains(position)].sample(1).Name.values[0]
                while check_player_in_lineup(lineup, random_player):
                    random_player = df[df['Position'].str.contains(position)].sample(1).Name.values[0]
                lineup.update({position: random_player})

        return lineup

    def get_fitness(self, lineup):
        data = self.dk_data
        filtered_lineup = data[data['Name'].isin(lineup.values())]

        if filtered_lineup.sum()['Salary'] > 50000:
            return -1

        lineup_score = 0
        # adds all columns in  config['rewards'] to calculate the lineup score
        for metric, weight in config['reward'].iteritems():
            lineup_score += filtered_lineup.sum()[metric] * weight

        self.fitness = lineup_score
        return self.fitness


class Population:

    def __init__(self, dk_data, desired_lineup):
        self.dk_data = dk_data
        self.desired_lineup = desired_lineup

        self.generation_num = 0
        self.current_generation = self.initialize_population()

        # logger.debug('Initial Population: {}'.format([individual.lineup for individual in self.current_generation]))

    def initialize_population(self):
        population = []
        for i in range(config['population_size']):
            population.append(Individual(self.dk_data, self.desired_lineup))
        return population

    def selection(self, number_of_top_individuals=config['select_top_individuals']):
        """
        According to the selection method select N chromosomes from the population from config['select_top_individuals']
        Returns list of individuals

        """
        selected_id_list = sorted(range(len(self.current_generation)), key=lambda i: self.current_generation[i].fitness)[- number_of_top_individuals:]
        selected_population = []
        for id in selected_id_list:
            selected_population.append(self.current_generation[id])
        return selected_population

    def crossover(self, parent1, parent2, cross_over_point):
        """
        Perform crossover on N chromosomes selected.
        Return tuple of 2 Class Individual that are a crossover of the 2 parents
        """

        # TODO add more cross over children
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        for i, (key, value) in enumerate(child1.lineup.iteritems()):
            if i >= cross_over_point:
                child1.lineup[key] = parent2.lineup[key]
                child2.lineup[key] = parent1.lineup[key]

        # Updates fitness value of children
        child1.get_fitness(child1.lineup)
        child2.get_fitness(child2.lineup)

        return child1, child2

    def mutation(self, child, rand_value=None):
        """
        Perform mutation on the chromosomes obtained.

        """
        df = self.dk_data

        if rand_value:
            rand_number = rand_value
        else:
            rand_number = np.random.uniform(low=0.0, high=1.0)

        if rand_number < config['mutate_probability']:
            logger.debug('Mutated Child!')
            random_position = random.sample(child.lineup, 1)
            random_position = random_position[0]
            logger.debug('random_position: {}'.format(random_position))
            logger.debug('old_lineup: {}'.format(child.lineup))

            if random_position == 'Util':
                random_player = df.sample(1).Name.values[0]
                while check_player_in_lineup(child.lineup, random_player):
                    random_player = df.sample(1).Name.values[0]
                child.lineup[random_position] = random_player
            else:
                random_player = df[df['Position'].str.contains(random_position)].sample(1).Name.values[0]
                while check_player_in_lineup(child.lineup, random_player):
                    random_player = df[df['Position'].str.contains(random_position)].sample(1).Name.values[0]
                child.lineup[random_position] = random_player

            logger.debug('new_lineup: {}'.format(child.lineup))
            logger.debug('mutated child - position: {} lineup: {}'.format(random_position, child.lineup))

        # update fitness of child
        child.get_fitness(child.lineup)
        return child


    def next_generation(self):
        """
        Perform selection, crossover, mutation, and replacement of population
        When end condition is satisfied return best solution from the population
        """
        # TODO add check to config file config['selected_top_individuals'] should be even and greater or equal to 2
        top_population = self.selection()
        new_population = []
        while top_population:
            parent1 = top_population.pop()
            parent2 = top_population.pop()

            # create random cross over point
            cross_over_point = random.randint(1, len(parent1.lineup)-2)
            # logger.debug('cross_over_point: {}'.format(cross_over_point))

            child1, child2 = self.crossover(parent1, parent2, cross_over_point)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            new_population.append(parent1)
            new_population.append(parent2)
            new_population.append(child1)
            new_population.append(child2)

        logger.debug('New Population Size: {}'.format(len(new_population)))
        self.current_generation = new_population
        return self.current_generation

    def evolve(self):
        while self.generation_num < config['generations']:

            print 'Generation Number: {}, Best Fitness: {}'.format(self.generation_num, self.selection(1)[0].fitness)
            logger.debug('Generation Number: {}, Best Fitness: {}'.format(self.generation_num, self.selection(1)[0].fitness))
            # logger.debug('generation: {}, lineup: {}'.format(self.generation_num, [individual.lineup for individual in self.current_generation]))
            self.next_generation()
            self.generation_num += 1

        top_lineup = self.selection(1)[0]
        return top_lineup


if __name__ == '__main__':
    logger.info('___________')
    logger.info('Script Start')

    if config['sport'] == 'debug':
        desired_lineup = ['PG', 'C', 'SG', 'Util']
        DK_SALARIES_FILE = 'Input/DKSalaries_Debug.csv'

    elif config['sport'] == 'nba':
        desired_lineup = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'Util']
    elif config['sport'] == 'mlb':
        desired_lineup = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    elif config['sport'] == 'pga':
        desired_lineup = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']

    dk_data = import_draftkings_salaries(DK_SALARIES_FILE)

    my_population = Population(dk_data, desired_lineup)
    best_lineup = my_population.evolve()

    logger.debug('Best Lineup: {}, Fitness: {}'.format(best_lineup.lineup, best_lineup.fitness))
    print('Best Lineup: {}, Fitness: {}'.format(best_lineup.lineup, best_lineup.fitness))

    # logger.info('DK Salaries: {}'.format(dk_data))

    logger.info('Script End\n')
