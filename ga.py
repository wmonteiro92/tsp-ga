# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:23:32 2020

@author: wmont
"""

# based on lecture IS 07, slide 25

import numpy as np
import pandas as pd
import ga_functions

# parameters
n_pop = 50
max_iter = 10000
crossover_size = 5

# 4 combinations of crossover and mutation rates
crossover_rates = [0.1, 0.35, 0.65, 1.0]
mutation_rates = [0.1, 0.35, 0.65, 1.0]

# creating a DataFrame to store the data
stats = []

# the benchmarks to be used
files = ['brazil58_matrix.out', 'swiss42_matrix.out', 'bays29_matrix.out']

# iterating through all the benchmark files
for file in files:
    # matrix is a city-location matrix
    matrix = np.loadtxt(file, delimiter=',')
    
    # initialize population with n_pop chromosomes
    pop = ga_functions.initialize_population(n_pop, len(matrix))
    
    # 30 runs for each combination of crossover and mutation rates
    for crossover_rate in crossover_rates:
        for mutation_rate in mutation_rates:
            # creating a empty list to store all the results found for this combination
            fitness = []
            for i in range(30):
                print(f'Run {i+1} of 30 - file {file}, ' + 
                      f'crossover {crossover_rate}, mutation {mutation_rate}.')
                
                # running the GA process
                curr_fitness, pop = ga_functions.run_ga(matrix, pop, n_pop,
                                        max_iter, crossover_size, crossover_rate,
                                        mutation_rate, debug=False)
                
                # appending the results of the current run to the fitness list
                fitness = fitness + curr_fitness
            
            # getting the results out of the current combination
            stats.append([file, crossover_rate, mutation_rate,
                             np.min(fitness), np.max(fitness), np.mean(fitness),
                             np.median(fitness), np.std(fitness)])
           
# creating a DataFrame (an easily manageable table) containing all the results
df_stats = pd.DataFrame(stats, columns=['Benchmark', 'CrossoverRate',
                    'MutationRate', 'Min', 'Max', 'Mean', 'Median', 'StdDev'])