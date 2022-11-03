import random
from statistics import mean
import math



class GA:

    def __init__(self):
        self.phase = "GENETIC ALGORITHM"

        # Individuals
        self.n_size = 50 # gene size (not used)
        self.grid = [[0,1]]*50 # list of list of size self.n_size, self.grid[i] defines the list of possible values for the i-th gene
        # Requirement: in self.grid, each element in the list must constain at least 2 elements

        # Population of individuals
        self.pop_size = 200 # population size (at the end of each iteration, the population size is self.pop_size)
        self.population = set() # population (is updated during every iteration)
        self.population_history = dict() # individual:fitness dictionary. self.population_history.keys() is a list of all individuals already encountered
        self.new_individuals = set() # new individuals added to the population for the next iteration (updated at every iteration)
        self.best_individual = None
        self.individuals_iter = dict()
        self.num_selected = 0 # number of new individuals selected

        # Fitness
        self.min_fitness = 0.01 # in order to define a strictly positive evaluation
        # self.f = lambda t: max(self.min_fitness*(1+1/(100-sum([-(i-25)**2 + 10 for i in range(len(t)) if t[i] == 1]))), 1/10*sum([-(i-25)**2 + 10 for i in range(len(t)) if t[i] == 1]))
        self.average_fitness = 0
        self.best_average_fitness = 0
        self.max_fitness = 0

        # Algorithm
        self.iterations = 50 # Stopping criterion (number of iterations)
        self.early_stopping = 5 # Stopping criterion (Number of iterations without increasing the average fitness of individuals in the population)
        self.stop_early_counter = 0

        # Mutation
        self.mutation_prob = 0.05 # Probability for a gene to be mutated during the mutation phase

    def reset(self):
        self.population = set() 
        self.population_history = dict()
        self.new_individuals = set() 
        self.best_individual = None
        self.individuals_iter = dict()
        self.average_fitness = 0
        self.best_average_fitness = 0
        self.max_fitness = 0
        self.stop_early_counter = 0

    def generate_initial_population(self):

        while len(self.population) < self.pop_size:
            tmp = tuple([random.choice(j) for j in self.grid])
            self.population.add(tmp)
            self.new_individuals.add(tmp) 


    def evaluate_new_individuals(self):
        f = self.fitness()
        for t in self.new_individuals:
            self.population_history[t] = f[t]
        self.new_individuals = set()
        fitnesses = [self.population_history[t] for t in list(self.population)] # ne pas recalculer celles précédemment calculées
        self.average_fitness = mean(fitnesses)
        self.max_fitness = max(fitnesses)
        # update the counter for early stopping
        if self.average_fitness > self.best_average_fitness:
            self.stop_early_counter = 0
            self.best_average_fitness = self.average_fitness
        else:
            self.stop_early_counter += 1
        return None

    def fitness(self): # to be redefinesd
        f = dict()
        # fitness = lambda t: max(self.min_fitness*(1+1/(100-sum([-(i-25)**2 + 10 for i in range(len(t)) if t[i] == 1]))), 1/10*sum([-(i-25)**2 + 10 for i in range(len(t)) if t[i] == 1]))
        fitness = lambda t: sum([((-1)**(i+1))*(i+1)*t[i] for i in range(len(t))])
        for t in self.new_individuals:
            f[t] = fitness(t)
        return f


    def select_best(self):
        # Rule: select the best individuals of the population such that their fitness
        # is higher than the average of the fitnesses calculated on the whole population 
        # (and within the limit, in number, of half the individuals of the population) 

        pop = list(self.population)
        fitnesses = [self.population_history[t] for t in pop] # ne pas recalculer celles précédemment calculées

        selected = sorted([(a,b) for a,b in zip(fitnesses, pop) if a > self.average_fitness], reverse = True)
        # At least, we need to keep 2 individuals
        if len(selected) == 1:
            selected = sorted([(a,b) for a,b in zip(fitnesses, pop)], reverse = True)[:2]
        if len(selected) >= math.ceil(0.5*self.pop_size): 
            selected = selected[:math.ceil(0.5*self.pop_size)]
        # Update population ()
        self.population = set([a for _,a in selected])
        self.num_selected = len(self.population)
        return None

    def crossover(self, t1, t2): # can be improved
        # select 2 pivots where to cut individuals
        n_size = len(t1)
        tmp = [i for i in range(n_size)]
        random.shuffle(tmp)
        cut1, cut2 = min(tmp[:2]), max(tmp[:2])
        # do a crossover between t1 and t2
        t1_, t2_ = [], []
        for i in range(n_size):
            if i < cut1 or i >= cut2:
                t1_.append(t1[i])
                t2_.append(t2[i])
            else:
                t1_.append(t2[i])
                t2_.append(t1[i])
        return(tuple(t1_), tuple(t2_))

    def mutate(self, t): 
        t_ = []
        for i in range(len(t)):
            if random.random() < self.mutation_prob:
                new_val = random.choice([elt for elt in self.grid[i] if not elt == t[i]])
                t_.append(new_val)  
            else: 
                t_.append(t[i])
        return(tuple(t_))


    def update_pop(self):
        pop_size_before = len(self.population)
        fitnesses_pop = [(self.population_history[t], t) for t in self.population] # list of (fitness, individuals) couples
        fitnesses = [f for f, _ in fitnesses_pop] 
        pop_selected = [p for _, p in fitnesses_pop]
        # Defines a distribution over the individuals in pop_selected
        # In the selection phase of the individuals to be crossed, the individuals with higher
        # fitness have more chance to be selected than the individuals with low fitness
        # Reminder:  fitness is a strictly positive evaluation function (f(t) > 0 for all t)
        fit_prob = [i/sum(fitnesses) for i in fitnesses]

        while len(self.population) < self.pop_size:
            # Select 2 parents
            r1 = random.random()
            r2 = random.random()
            cum = 0
            parent1, parent2 = None, None
            for i in range(pop_size_before):
                cum += fit_prob[i]
                if r1 < cum and parent1 == None:
                    parent1 = pop_selected[i]
                if r2 < cum and parent2 == None:
                    parent2 = pop_selected[i]
                if not parent1 == None and not parent2 == None:
                    break
            
            # Make them reproduce (don't forget to add some genetic diversity 
            # via the gene mutation principle)
            individuals_encountered = set(self.population_history.keys())
            if not parent1 == parent2:
                t1_, t2_ =  self.crossover(parent1, parent2) 
                t1_ = self.mutate(t1_)
                t2_ = self.mutate(t2_)
                # Add individuals to the population if they are not carbon 
                # copies of individuals already encountered in the past
                if not t1_ in individuals_encountered:
                    self.population.add(t1_)
                    self.new_individuals.add(t1_)
                    len(self.population)
                if not len(self.population) == self.pop_size and not t2_ in individuals_encountered:
                    self.population.add(t2_)
                    self.new_individuals.add(t2_)
                    len(self.population)

        return None

    def update_individuals_iter(self, i):
        for t in self.new_individuals:
            self.individuals_iter[t] = i

    def print_100_best(self):
        # Print the 100 best individuals
        # If less than 100 individuals in population history, print all individuals
        r = sorted([(self.population_history[t], t) for t in self.population_history.keys()], reverse=True)
        for i in range(min(len(r), 100)):
            # reports.print_report_line(self.phase, 'First best individuals', {"rank": i, "fitness": r[i][0], "individual": r[i][1]})
            print(f'First best individuals -- rank: {i}, fitness: {r[i][0]}, individual: {r[i][1]}')

    def print_all(self):
        r = sorted([(self.population_history[t], self.individuals_iter[t], t) for t in self.population_history.keys()], reverse=True)
        for i in range(len(r)):
            reports.print_report_line(self.phase, 'First best individuals', {"rank": i, "fitness": r[i][0], "iteration": r[i][1], "individual": r[i][2]})

    def algo(self):
        self.generate_initial_population()
        self.update_individuals_iter(0)
        self.evaluate_new_individuals()
        # reports.print_report_line(self.phase, 'End of first generation', {"iteration": 0, "fitness max": self.max_fitness, "fitness avg": self.average_fitness, "best individuals selected": self.num_selected, "new individuals": self.pop_size-self.num_selected, "total individuals evaluated": len(self.population_history.keys())})
        for i in range(self.iterations):
            self.select_best() # reduce the size of the population to the best elements
            # reports.print_report_line(self.phase, 'Best individuals selected', {"number": len(self.population)})
            # print(f'Number of best individuals selected: {len(self.population)}')
            self.update_pop() # add new individuals to the population (via crossover and mutation)
            # print(f'Population size: {len(self.population)}')
            # reports.print_report_line(self.phase, 'New individuals in the population', {"number": len(self.new_individuals)})
            # print(f'New_individuals: {len(self.new_individuals)}')
            self.update_individuals_iter(i+1)
            self.evaluate_new_individuals()
            # reports.print_report_line(self.phase, 'Total individuals evaluated in history',{"number": len(self.population_history.keys())} )

            # reports.print_report_line(self.phase, 'End of a generation', {"iteration": i+1, "fitness max": self.max_fitness, "fitness avg": self.average_fitness, "best individuals selected": self.num_selected, "new individuals": self.pop_size-self.num_selected, "total individuals evaluated": len(self.population_history.keys())})
            # print(f'Number of individuals encountered: {len(self.population_history.keys())}')
            # if self.stop_early_counter == self.early_stopping:
            #     reports.print_report_line(self.phase, 'Early stopping', {"iteration": i+1})
            #     # print(f'\nEarly stopping after {i+1} iterations')
            #     break
        r = sorted([(self.population_history[t], t) for t in self.population], reverse=True)
        self.best_individual = r[0][1]
        self.max_fitness = self.population_history[self.best_individual]

        # reports.print_report_line(self.phase, "Results", {"fitness max": self.max_fitness, "fount at iteration": self.individuals_iter[self.best_individual], "best individual": self.best_individual})
        # print(f'end algo gen. Fitness max: {self.max_fitness}, found at iteration: {self.individuals_iter[self.best_individual]}, best individual: {self.best_individual}')

        # print(f'RESULTS: fitness max = {self.max_fitness}, best_individual = {self.best_individual}, found at iteration = {self.individuals_iter[self.best_individual]}')


if __name__ == "__main__":
    # default genetic algorithm tries to find the maximum of a simple function
    # f(x1,x2, ..., x50) = sum_i [(-1)^i * xi * i]
    # with xi in {0, 1} for all i
    # The best and unique solution is (x1, x2, x3, x4, ..., x47, x48, x49, x50) = (0, 1, 0, 1, ..., 0, 1, 0, 1)
    # And the maximum is 650
    random.seed(42)
    ga = GA()
    search_space_size = math.prod([len(i) for i in ga.grid])
    print(f"Search space size: {search_space_size} individuals")
    ga.algo()
    print(f"Individuals evaluated: {len(ga.population_history.keys())} individuals")
    print(f'Maximum found: {ga.max_fitness}')
    print(f'Best individual: {ga.best_individual}')