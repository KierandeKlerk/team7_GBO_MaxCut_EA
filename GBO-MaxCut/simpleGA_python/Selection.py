import numpy as np

def select_best_solution( candidates ):
	best_ind = np.argmax([ind.fitness for ind in candidates])
	return candidates[best_ind]

def tournament_selection( population, offspring ): 
	selection_pool = np.concatenate((population, offspring),axis=None)
	tournament_size = 4
	assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of 2"
	
	selection = []
	number_of_rounds = tournament_size//2
	for i in range(number_of_rounds):
		number_of_tournaments = len(selection_pool)//tournament_size
		order = np.random.permutation(len(selection_pool))
		for j in range(number_of_tournaments):
			indices = order[tournament_size*j:tournament_size*(j+1)]
			best = select_best_solution(selection_pool[indices])
			selection.append(best)
	assert( len(selection) == len(population) )

	return selection
