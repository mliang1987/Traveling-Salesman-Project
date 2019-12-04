import util as ut
import numpy as np
import simulated_annealing as sa
import random
import pandas as pd
import time
import math


class GeneticAlgorithm():

	def __init__(self, name,  coordinates, randomSeed = 0, pop_size = 300, max_iteration_time = 60, num_crossovers = 140, num_mutation = 250, test_quality = None):
		'''
		Constructor for Genetic Algorithms.

		Parameters:
		name: String for the name of the file
		coordinates: dictionary of node IDs to coordinates.

		Optional:
		randomSeed: random seed for Python random number generator
		pop_size: population size of the sample
		max_iteration_time: maximum time in seconds before we stop 
		num_crossovers: number of crossovers from one population to the next
		num_mutation: number of mutations one individuals on each population
		'''

		# Problem parameters
		self.coordinates = coordinates
		self.name = name
		self.N = len(coordinates)
		self.nodes = list(self.coordinates.keys())
		self.max_iteration_time = max_iteration_time
		# Seed random number generator
		self.randomSeed = randomSeed
		random.seed(randomSeed)

		#GA parameters
		self.pop_size = pop_size
		self.num_crossovers = num_crossovers
		self.num_mutation = num_mutation
		self.trace = []

		if test_quality == None:
			self.ideal = 1.0/7542

		else:
			self.ideal = 1.0/test_quality


	def distance(self, n1, n2):
		'''
		Calculates the distances between nodes.

		Parameters:
		n1: Node 1 ID
		n2: Node 2 ID

		Returns: floating point Euclidean distance between two nodes
		'''
		x1,y1 = self.coordinates[n1]
		x2,y2 = self.coordinates[n2]
		return math.sqrt((x1-x2)**2 +(y1-y2)**2)
		
	def random_tour(self):
		'''
		Random tour generated.  Identifies all nodes, then shuffles order.

		Returns:
		solution: tour path
		'''
		path = list(self.coordinates.keys())
		random.shuffle(path)
		return path


	def nearest_neighbors_tour(self, start_city):
		'''
		Tour generated with greedy heuristic: Picks a random starting node, then appends
		next nearest neighbor.

		Returns:
		solution: tour path
		'''
		solution = []
		unassigned_nodes = set(self.nodes)
		node = start_city
		solution.append(node)
		unassigned_nodes.remove(node)
		while unassigned_nodes:
			next = min(unassigned_nodes, key=lambda x: self.distance(node, x))
			unassigned_nodes.remove(next)
			solution.append(next)
			node = next
		return solution

	def terminate(self, count):
		'''
		Returns True if we have reached the time limit assigned to Genetic Algorithm for 
		current instaance.

		Returns:
		Boolean representing whether we have reached cutoff time or not
		'''
		if count>= self.max_iteration_time:
			return True
		return False

	def selectIndividualsForMutation(self, population):
		'''
		Randomly select required number of individuals for mutations

		Returns:
		population_mutation: List of individuals selected for mutation
		'''
		n = self.num_mutation
		pop = np.arange(self.pop_size)
		selected = np.random.choice(pop, size = n, replace = False)
		population_np = np.array(population)
		population_mutation = population_np[selected][:].tolist()
		return population_mutation

	def pickIndividualsForCrossover(self, population):
		'''
		Select individuals for crossover, using fitness measure of an individual 
		as its probability distribution for being selected.

		Returns:
		population_cross: List of individuals selected for crossover
		'''		
		all_fitness = self.populationFitness(population)
		fitness = np.array(all_fitness)
		cum_fitness = np.cumsum(fitness)
		cum_fitness = (cum_fitness*100)/np.sum(fitness)
		n = 2*self.num_crossovers
		population_cross = []
		for i in range(n):
			r = random.random()*100
			for j in range(len(cum_fitness)):
				if r<= cum_fitness[j]:
					population_cross.append(population[j])
		return population_cross

	def globalCompetition(self, oldPopulation, newPopulation):
		'''
		Select the (pop_size) fittest individuals from oldPopulation and newPopulation combined

		Returns:
		finalpopulation: list containing fittest individuals for new population
		'''
		oP = sorted(oldPopulation, key = self.evaluateFitness, reverse = True)
		nP = sorted(newPopulation, key = self.evaluateFitness, reverse = True)
		finalPopulation = [[0 for _ in range(self.N)] for _ in range(self.pop_size)]
		j = 0
		k=0
		i=0
		while i < self.pop_size and j<len(oP) and k< len(nP):
			if self.evaluateFitness(oP[j]) > self.evaluateFitness(nP[k]):
				finalPopulation[i] = oP[j]
				j+=1
			else:
				finalPopulation[i] = nP[k]
				k+=1
			i+=1
		while i < self.pop_size and j<len(oP) :
			finalPopulation[i] = oP[j]
			j+=1
			i+=1

		while i < self.pop_size and  k< len(nP):
			finalPopulation[i] = nP[k]
			k+=1
			i+=1

		return finalPopulation

	def getBestIndividual(self,population):
		'''
		Finds the fittest individual in current population

		Returns:
		best: fittest individual in the population
		bestFitness: value of highest fitness
		'''
		bestFitness = float('-inf')
		best = []
		for i in range(self.pop_size):
			currentFitness = self.evaluateFitness(population[i])
			if  currentFitness > bestFitness:
				best = population[i]
				bestFitness = currentFitness
		return best, bestFitness

	def populationFitness(self, population):
		'''
		Computes and returns fitness of the whole population

		Returns:
		all_fitness: List of fitness value of each individual
		'''
		all_fitness = []
		for i in range(self.pop_size):
			all_fitness.append(self.evaluateFitness(population[i]))
		return all_fitness

	def evaluateFitness(self, current):
		'''
		Computes fitness of an individual

		Returns:
		1/path_value: the fitness of an individual
		'''
		fitness = ut.get_tour_distance(current, self.coordinates)
		return (1.0/fitness)

	def mutate(self, current):
		'''
		Given an individual, make a random mutation

		Returns:
		mutant: mutated path
		'''
		index1 = random.randint(0,self.N-1)
		index2 = random.randint(0,self.N-1)
		mutant = current.copy()
		mutant[index1], mutant[index2] = mutant[index2], mutant[index1]
		return mutant

	def crossover(self,mate1, mate2):
		'''
		Given 2 individuals/paths, compute a crossover to produce 2 offsprings

		Returns:
		offspring1: Reuslt of mutation between the 2 individuals
		offspring2: Reuslt of mutation between the 2 individuals
		'''
		index1 = random.randint(0,self.N-1)
		index2 = random.randint(0,self.N-1)
		while index2 == index1:
			index2 = random.randint(0,self.N-1)
		if index2<index1:
			index1,index2 = index2, index1

		offspring1 = mate1.copy()
		offspring2 = mate2.copy()

		count = {}
		for i in range(index1, index2+1):
			count[offspring1[i]] = True

		i = index2+1
		j = index2+1
		while (j%self.N)!=(index1):
			if mate2[i%self.N] not in count:
				offspring1[j%self.N] = mate2[i%self.N]
				j+=1
			i+=1

		count = {}
		for i in range(index1, index2+1):
			count[offspring2[i]] = True

		i = index2+1
		j = index2+1
		while (j%self.N)!=(index1):
			if mate1[i%self.N] not in count:
				offspring2[j%self.N] = mate1[i%self.N]
				j+=1
			i+=1

		return offspring1, offspring2


	def GeneticAlgo(self):
		'''
		Genetic Algorithm process:
		1. Generate initial population using a combination of All nearest neighbors and random paths
		2. Keep mutating, crossing over individuals from current population to a new population
		3. Select the fittest individuals from the current and new population to a new current population
		4. Repeat till we reach cutoof time or get ideal solution
		
		Returns:
		Best Inidvidual(Path) in Final Population
		'''
		start_time = time.time()
		pop_size = self.pop_size
		solution = []
		population = [[0 for _ in range(self.N)] for _ in range(pop_size)]

		##initial population using all nearest neighbors
		for i in range(self.N):
			population[i] = self.nearest_neighbors_tour(self.nodes[i])

		##use random tour for remaining initial population
		for i in range(pop_size-self.N):
			population[i+self.N] = self.random_tour()


		prevBestFitness = float("-inf")


		## while time<cutoff time
		while(self.terminate(time.time()-start_time)!= True):
			newPopulation = []
			bestI, bestFitness = self.getBestIndividual(population)
			## write to the trace file
			if bestFitness> prevBestFitness:
				self.trace.append([round((time.time()-start_time),2),ut.get_tour_distance(bestI, self.coordinates)])

			prevBestFitness = bestFitness

			if bestFitness >= self.ideal:
				break



			## get a mutation pool, then get mutants
			mutation_pool = self.selectIndividualsForMutation(population)
			for i in range(len(mutation_pool)):
				parent = mutation_pool[i]
				mutant = self.mutate(parent)
				newPopulation.append(mutant)

			## get a crossover pool, then get offsprings
			crossover_pool = self.pickIndividualsForCrossover(population)
			for i in range(self.num_crossovers):
				parent1 = crossover_pool[2*i]
				parent2 = crossover_pool[(2*i)+1]
				offspring1, offspring2 = self.crossover(parent1, parent2)
				newPopulation.append(offspring1)
				newPopulation.append(offspring2)

			## get the final population, by computing fittest individuals from population+newpopulation
			population = self.globalCompetition(population, newPopulation)

		return self.getBestIndividual(population)





def ga_single(file_path, max_time, random_seed = 0,  test_quality = None):
	coordinates = ut.read_tsp_file(file_path)
	ga = GeneticAlgorithm(file_path, coordinates, randomSeed = random_seed, max_iteration_time = max_time,  test_quality = test_quality)

	result,_ = ga.GeneticAlgo()
	cost = ut.get_tour_distance(result, coordinates)
	trace = ga.trace
	return cost,[x + 1 for x in result],trace



	

