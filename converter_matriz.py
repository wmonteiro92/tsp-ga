# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:40:09 2020

@author: wmont
"""
import numpy as np

def ler_arquivo(file, start_line, end_line):
    lines = open(f'{file}.tsp', 'r').read().splitlines()[start_line:end_line]
    values = [[int(value) for value in line.split()] for line in lines]
    return values

def converter_matrix_triangular(values):
    values = [[[0] * (len(values) - len(line) + 1) + line for line in values]][0]
    values.append([0] * len(values[0]))
    
    # criando a matriz simétrica
    matrix = np.triu(values)
    return matrix + matrix.T

# lendo o arquivo
file = 'brazil58'
values = ler_arquivo(file, 7, -1)
np.savetxt(f'{file}_matrix.out', converter_matrix_triangular(values), delimiter=',')

# lendo o arquivo
file = 'bays29'
values = ler_arquivo(file, 8, 37)
np.savetxt(f'{file}_matrix.out', np.matrix(values), delimiter=',')

# lendo o arquivo
file = 'swiss42'
values = ler_arquivo(file, 7, 49)
np.savetxt(f'{file}_matrix.out', np.matrix(values), delimiter=',')