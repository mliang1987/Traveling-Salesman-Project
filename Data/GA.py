import util as ut
import numpy as np
import simulated_annealing as sa
import random
import pandas as pd
import time


class GeneticAlgorithm():

	def __init__(self, name, coordinates, pop_size = 400, max_iteration_time = 600, num_crossovers = 180, num_mutation = 400):
		self.coordinates = coordinates
		self.name = name
		self.N = len(coordinates)
		self.nodes = list(self.coordinates.keys())

		self.max_iteration_time = max_iteration_time
		self.pop_size = pop_size
		self.num_crossovers = num_crossovers
		self.num_mutation = num_mutation

		df = pd.read_csv('C:/Users/LBJ/Desktop/CANVAS/Algo/CSE-6140---Project-master/Data/solutions.csv', header = 0)
		df.set_index("Instance", inplace=True)
		self.ideal = 1.0/(df.loc[self.name]['Value'])


	def random_tour(self):
		'''
		Random tour generated.  Identifies all nodes, then shuffles order.

		Returns:
		solution: tour path
		fitness: tour length
		'''
		path = list(self.coordinates.keys())
		random.shuffle(path)
		#fitness = ut.get_tour_distance(path, self.coordinates)
		return path#, fitness

	def terminate(self, count):
		if count>= self.max_iteration_time:
			return True
		return False

	def selectIndividualsForMutation(self, population):
		n = self.num_mutation
		pop = np.arange(self.pop_size)
		selected = np.random.choice(pop, size = n, replace = False)
		population_np = np.array(population)
		population_mutation = population_np[selected][:].tolist()
		return population_mutation

	def pickIndividualsForCrossover(self, population):
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
		# pop = np.arange(self.pop_size)
		# selected = np.random.choice(pop, size = n, replace = False)
		# population_np = np.array(population)
		# population_cross = population_np[selected][:].tolist()
		return population_cross

	def globalCompetition(self, oldPopulation, newPopulation):
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
		bestFitness = float('-inf')
		best = []
		for i in range(self.pop_size):
			currentFitness = self.evaluateFitness(population[i])
			if  currentFitness > bestFitness:
				best = population[i]
				bestFitness = currentFitness
		return best, bestFitness

	def populationFitness(self, population):
		all_fitness = []
		for i in range(self.pop_size):
			all_fitness.append(self.evaluateFitness(population[i]))
		return all_fitness

	def evaluateFitness(self, current):
		#print(current)
		fitness = ut.get_tour_distance(current, self.coordinates)
		return (1.0/fitness)

	def mutate(self, current):
		index1 = random.randint(0,self.N-1)
		index2 = random.randint(0,self.N-1)
		mutant = current.copy()
		mutant[index1], mutant[index2] = mutant[index2], mutant[index1]
		return mutant

	def crossover(self,mate1, mate2):
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
		pop_size = self.pop_size
		solution = []
		population = [[0 for _ in range(self.N)] for _ in range(pop_size)]
		for i in range(pop_size):
			population[i] = self.random_tour()

		start_time = time.time()
		#count =0

		while(self.terminate(time.time()-start_time)!= True):
			newPopulation = []
			_, bestFitness = self.getBestIndividual(population)
			if bestFitness == self.ideal:
				break
			#print('Iteration ', count, ' Best Fitness = ', bestFitness )
			mutation_pool = self.selectIndividualsForMutation(population)
			for i in range(len(mutation_pool)):
				parent = mutation_pool[i]
				mutant = self.mutate(parent)
				newPopulation.append(mutant)

			crossover_pool = self.pickIndividualsForCrossover(population)
			for i in range(self.num_crossovers):
				parent1 = crossover_pool[2*i]
				parent2 = crossover_pool[(2*i)+1]
				offspring1, offspring2 = self.crossover(parent1, parent2)
				newPopulation.append(offspring1)
				newPopulation.append(offspring2)

			#print(population)
			#print(newPopulation)
			population = self.globalCompetition(population, newPopulation)

			#count+=1

		return self.getBestIndividual(population)

def genetic_tests():
	'''
	Tests out simulated annealing algorithm using default parameters.
	'''
	all_coordinates = ut.get_all_files()
	for city, coordinates in all_coordinates.items():
		ga = GeneticAlgorithm(city, coordinates)
		result,_ = ga.GeneticAlgo()
		print("Results for {}:".format(city))
		ut.plotTSP(result, coordinates, title = "Genetic Algorithm: "+city, save_path = "C:/Users/LBJ/Desktop/CANVAS/Algo/CSE-6140---Project-master/Data/Plots/GA3/"+city+".png", verbose = True)
	pass

if __name__ == "__main__":
	genetic_tests()

	

