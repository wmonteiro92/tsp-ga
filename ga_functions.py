# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:49:23 2020

@author: wmont
"""

import numpy as np

def calculate_fitness(element, matrix):
    '''Calculates the fitness of a given element based on the distances informed in the matrix.'''
    distance = 0
    for i in range(len(element)-1):
        distance += matrix[element[i]][element[i+1]] 
    return distance

def initialize_population(n_pop, n_elements):
    '''Initializes a random population.'''
    population = [0]*n_pop

    list_elements = list(range(n_elements))
    for i in range(n_pop):
        # generating the element order
        element = list_elements.copy()
        np.random.shuffle(element)
        population[i] = element   
        
    return population

def random_selection(population, fitness):
    '''Randomly chooses elements out of the population.'''
    normalized_fitness = fitness / np.sum(fitness)
    
    ranking = np.argsort(normalized_fitness)
    normalized_fitness = np.sort(normalized_fitness)
    cumulative_sum = np.cumsum(normalized_fitness)
    
    # since we want to minimize the distances, we're using argmin
    chosen_element = np.argmin(cumulative_sum >= np.random.random())
    return population[ranking[chosen_element]]

def generate_cuts(elements):
    '''Generates the cuts for the crossover.'''
    cuts = [0, 0]
    cuts[0] = np.random.randint(elements-2)+2
    
    while True:
        cuts[1] = np.random.randint(elements-2)+2
        if cuts[1] != cuts[0]:
            break
        
    return cuts

def remove_elements(elements, elements_to_ignore):
    '''Eliminates items from the first list contained in the second list.'''
    return [element for element in elements if element not in
            elements_to_ignore]

def crossover(population, new_population, crossover_size, crossover_rate):
    '''Crossover step.'''
    # based on lecture IS 07, slide 33 (order crossover)
    offspring = []
    n_elements = len(population[0])
    
    cuts = np.random.randint(n_elements-crossover_size-1)
    cuts = [cuts, cuts+5]
    
    for i in range(len(population)):
        # generating the offspring based on their parents
        offspring_1 = population[i].copy()
        offspring_2 = new_population[i].copy()
        
        if (np.random.random() <= crossover_rate):
            # copying the bits between the cutting points
            cut_1 = population[i][cuts[0]:cuts[1]]
            cut_2 = new_population[i][cuts[0]:cuts[1]]
            
            # getting the bit sequence from the second cut onwards
            order_1 = remove_elements(new_population[i][
                    cuts[1]:] + new_population[i][:cuts[1]], cut_1)
            
            offspring_1[cuts[1]:] = order_1[:n_elements-cuts[1]]
            offspring_1[:cuts[0]] = order_1[n_elements-cuts[1]:]
            
            order_2 = remove_elements(population[i][
                    cuts[1]:] + population[i][:cuts[1]], cut_2)
            offspring_2[cuts[1]:] = order_2[:n_elements-cuts[1]]
            offspring_2[:cuts[0]] = order_2[n_elements-cuts[1]:]
        
        offspring.append(offspring_1)
        offspring.append(offspring_2)
    
    return offspring

def mutation(population, mutation_rate):
    '''Mutation step.'''
    # based on lecture IS 07, slide 40 (reciprocal exchange)
    offspring = []
    n_elements = len(population[0])
    
    for i in range(len(population)):
        child = population[i].copy()
        if (np.random.random() <= mutation_rate):
            first_element = np.random.randint(n_elements-2)
            second_element = np.random.randint(first_element, n_elements-1)
        
            child[first_element] = population[i][second_element]
            child[second_element] = population[i][first_element]
        
        offspring.append(child)
    
    return offspring

def run_ga(matrix, pop, n_pop, max_iter, crossover_size, crossover_rate,
           mutation_rate, debug):
    '''Genetic Algorithm for Traveling Salesman Problems (TSP).'''
    # current iteration
    current_iteration = 1
    
    # while the stopping criteria had not been met
    while current_iteration <= max_iter:
        # fitness calculation
        fitness = [0] * n_pop
        for i in range(len(pop)):
            fitness[i] = calculate_fitness(pop[i], matrix)
        
        new_pop = pop.copy()
        for i in range(len(pop)):
            # selection
            new_pop[i] = random_selection(pop, fitness)
            
        # crossover
        cross_pop = crossover(pop, new_pop, crossover_size, crossover_rate)
        
        # mutation
        mut_pop = mutation(cross_pop, mutation_rate)
        
        # fitness calculation
        fitness = [0] * len(mut_pop)
        for i in range(len(mut_pop)):
            fitness[i] = calculate_fitness(mut_pop[i], matrix)
        
        # pruning (limiting the number of elements by n_pop)
        ranking = np.argsort(fitness)[:n_pop]
        fitness = list(fitness[i] for i in ranking)
        pop = list(mut_pop[i] for i in ranking)
        
        # showing stats
        if debug:
            print(f'Iteration {current_iteration}, ' +
                  f'min. distance {np.min(fitness)}, ' +
                  f'max. distance {np.max(fitness)}.')
        
        # incrementing the counter
        current_iteration += 1
        
    return fitness, pop