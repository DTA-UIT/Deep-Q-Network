from __future__ import unicode_literals, print_function, division
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
from numpy import random 

def f(x, y):
	return .26*(x**2 + y**2) - .48*x*y

def gaussian_distribution(x_best, standard_derivation):
	x = np.array(random.normal(loc = x_best, scale = standard_derivation, size = (25, 1)))
	return x

def offspring(x_vector, y_vector, x_best, y_best, x_standard_deviation, y_standard_deviation):
	x_vector, y_vector = gaussian_distribution(x_best, x_standard_deviation), gaussian_distribution(y_best, y_standard_deviation)
	return x_vector, y_vector

def sort_generation(x_vector, y_vector, local_current_vector, x_mean_vector, y_mean_vector):
	local_current_vector_sorted = []
	x_mean_vector = []
	y_mean_vector = []

	for i in x_vector:
		for j in y_vector:
			local_current_vector_sorted.append(f(i, j))

	local_current_vector_sorted.sort()

	for a in range(mean_vector_size):
		for i in x_vector:
			for j in y_vector:
				if local_current_vector_sorted[a] == f(i, j):
					x_mean_vector.append(i)
					y_mean_vector.append(j)
	
	for i in range(mean_vector_size):
		local_current_vector.append(local_current_vector_sorted[i])

	return local_current_vector_sorted, x_mean_vector, y_mean_vector

def print_vector(x_vector):
	for i in x_vector:
		print(i, end = ' ')
	print('\n')

def update_deviation(x_vector, new_standard_deviation):
	for i in x_vector:
		new_standard_deviation += i 
	return new_standard_deviation/mean_vector_size

def compare_gene(x_vector, y_vector):
	for i in x_vector:
		for j in y_vector:
			if i != j:
				return False 
	return True

def convergence(local_curr_old, local_curr_new, counter):
    if local_curr_old == local_curr_new:
        counter+=1
    if counter == 10:
        return True

generation = 1000

size_vector = 50
maxx = 10000
counter = 0

x_initialize = 10
y_initialize = -10

x_best = x_initialize
y_best = y_initialize
x_best_minus = -x_initialize
y_best_minus = -y_initialize

mean_vector_size = 25

x_standard_deviation = 1
y_standard_deviation = 1
x_vector = np.array([])
y_vector = np.array([])

x_first_generation = gaussian_distribution(x_initialize, x_standard_deviation)
y_first_generation = gaussian_distribution(y_initialize, y_standard_deviation)

local_current_vector = []
x_vector = x_first_generation
y_vector = y_first_generation
x_mean_vector = []
y_mean_vector = []
local_minimum = local_current = f(x_vector[0], y_vector[0])

#initialize
local_current_vector, x_mean_vector, y_mean_vector = sort_generation(x_vector, y_vector, local_current_vector, x_mean_vector, y_mean_vector)
print('X - Ancestor: ')
print_vector(x_vector)
print('Y - Ancestor: ')
print_vector(y_vector)

x_sum_vector, y_sum_vector = 0, 0
for _ in x_mean_vector:
    x_sum_vector += _

for _ in y_mean_vector:
    y_sum_vector += _

for a in range(mean_vector_size):
	local_minimum_vector = f(x_vector[a], y_vector[a])

x_best, y_best = x_sum_vector/mean_vector_size, y_sum_vector/mean_vector_size
local_current_mean = f(x_best, y_best)

while convergence(local_current_mean, local_minimum, counter) is not True:

	print('Generation #' + str(generation) + ':')

	np.delete(x_mean_vector, 1, None)
	np.delete(y_mean_vector, 1, None)
	np.delete(x_vector, 1, None)
	np.delete(y_vector, 1, None)
	np.delete(local_current_vector, 1, None)

	#update standard deviation
	x_standard_deviation = abs(update_deviation(x_vector, x_standard_deviation))
	y_standard_deviation = abs(update_deviation(y_vector, y_standard_deviation))
	#update vector
	for a in range(size_vector):
		x_vector, y_vector = offspring(x_vector, y_vector, x_best, y_best, x_standard_deviation, y_standard_deviation)

	local_current_vector, x_mean_vector, y_mean_vector = sort_generation(x_vector, y_vector, local_current_vector, x_mean_vector, y_mean_vector)

	# print('X - parents: ')
	# print_vector(x_mean_vector)
	# print('Y - parents: ')	
	# print_vector(y_mean_vector)

	x_sum_vector = 0
	for i in range(mean_vector_size):
		x_sum_vector += x_mean_vector[i]

	x_best = x_sum_vector/mean_vector_size

	y_sum_vector = 0
	for i in range(mean_vector_size):
		y_sum_vector += y_mean_vector[i] 
	y_best = x_sum_vector/mean_vector_size

	local_current_mean = f(x_best, y_best)

	print('X_best: ')
	print_vector(x_best)
	print('Y_best: ')
	print_vector(y_best)
	print('Local current - Children: ')
	print_vector(local_current_mean)
 
	plt.show()


	if(local_current_mean < local_minimum):	
		local_minimum = local_current_mean
    
	generation += 1