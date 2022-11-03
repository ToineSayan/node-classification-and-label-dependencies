from genetic_algoritm import GA
import random

class GA_grid_search(GA):
    
    def __init__(self, grid):
        super().__init__()

        self.loaded_grid = grid
        # transform lists into string for hashability
        # LIST indicates a transformation, necessary for inverse transformation
        for k,v in self.loaded_grid.items():
            self.loaded_grid[k] = ["LIST" + str(e) if type(e) == list else e for e in v]
        self.grid_keys = list(self.loaded_grid.keys())
        self.grid = [self.loaded_grid[k] for k in self.grid_keys] 

        self.grid_size = 1
        for v in self.loaded_grid.values():
            self.grid_size = self.grid_size*len(v)

        self.iterations = 9 # = 10 iterations 
        self.pop_size = 100

        
    def generate_initial_population(self):
        # Select self.pop_size individuals randomly in the search space
        random_list = list(range(self.grid_size))
        random.shuffle(random_list)
        random_list = random_list[:self.pop_size]
        for random_int in random_list:
            config_n = dict()
            r, q = 0, random_int
            for k in self.grid_keys:
                r = q % len(self.loaded_grid[k])
                q = int((q - r)/len(self.loaded_grid[k]))
                config_n[k] = self.loaded_grid[k][r]
            elt_ = tuple(list(config_n.values()))

            self.population.add(elt_)
            self.new_individuals.add(elt_)
        return None

    def fitness_individual(self, individual):
        # TO BE REDEFINED
        print("WARNING: fitness_individual method has not been instanciated")
        return 1

    def fitness(self):
        r = dict()
        for conf in self.new_individuals:
            conf_fitness = self.fitness_individual(conf)
            r[conf] = conf_fitness
        return r
     
    def reset(self):
        super().reset()

        
class Full_grid_search(GA_grid_search):
    def __init__(self, configuration, jean_zay = False):
        super().__init__(configuration, jean_zay)
        self.pop_size = self.Conf.number_of_configs
        self.phase = 'FULL GRID SEARCH'
    
    def reset(self):
        super().reset()
        self.pop_size = self.Conf.number_of_configs
    
    def algo(self):
        self.generate_initial_population()
        self.update_individuals_iter(0)
        self.evaluate_new_individuals()
        r = sorted([(self.population_history[t], t) for t in self.population], reverse=True)
        self.best_individual = r[0][1]
        self.max_fitness = self.population_history[self.best_individual]






