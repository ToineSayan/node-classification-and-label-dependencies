import os
import json
from genetic_algoritm import GA
import random
import time
from math import ceil 
import copy

# from NOPAR_config_default import get_default_config
# from NOPAR_GMNN import GMNN




class GA_grid_search(GA):
    
    def __init__(self, grid):
        super().__init__()

        self.loaded_grid = grid
        for k,v in self.loaded_grid.items():
            self.loaded_grid[k] = ["LIST" + str(e) if type(e) == list else e for e in v]
        self.grid_keys = list(self.loaded_grid.keys())
        self.grid = [self.loaded_grid[k] for k in self.grid_keys] 

        self.grid_size = 1
        for v in self.loaded_grid.values():
            self.grid_size = self.grid_size*len(v)

        self.seed = 42 
        random.seed(self.seed)
        self.iterations = 9 # = 10 iterations 
        self.pop_size = 100
        # self.iterations = 1 # = 10 iterations 
        # self.pop_size = 5
        
    def generate_initial_population(self):


        # Select 100 individuals randomly
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
            # transform lists into string for hashability
            # LIST indicates a transformation, necessary for inverse transformation
            # tmp = list(config_n.values())
            # tmp = ["LIST" + str(e) if type(e) == list else e for e in tmp]
            elt_ = tuple(list(config_n.values()))

            # print(elt_)
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
        reports.print_report_line(self.phase, 'End of first generation', {"iteration": 0, "fitness max": self.max_fitness, "fitness avg": self.average_fitness, "best individuals selected": self.num_selected, "new individuals": self.pop_size-self.num_selected, "total individuals evaluated": len(self.population_history.keys())})
        r = sorted([(self.population_history[t], t) for t in self.population], reverse=True)
        self.best_individual = r[0][1]
        self.max_fitness = self.population_history[self.best_individual]
        reports.print_report_line(self.phase, "Results", {"fitness max": self.max_fitness, "fount at iteration": self.individuals_iter[self.best_individual], "best individual": self.best_individual})



if __name__ == "__main__":


    import sys, copy
    import ast
    import argparse
    import torch
    import json
    import load_datasets
    from split import train_val_test_split
    from trainer import Trainer
    import early_stopping

    from models.gcn import GCN
    from models.perceptron import MLP 
    from models.graphSAGE import GraphSAGE
    from models.fagcn import FAGCN

    from statistics import mean, stdev


    # for (dataset_name, model, split_suffix) in [
    #     ('CORA-ORIG', 'MLP', 's'),
    #     ('CORA-ORIG', 'GCN', 's'),
    #     ('CORA-ORIG', 'FAGCN', 's'),
    #     ('CITESEER-ORIG', 'MLP', 's'),
    #     ('CITESEER-ORIG', 'GCN', 's'),
    #     ('CITESEER-ORIG', 'FAGCN', 's'),
    #     ('PUBMED-ORIG', 'MLP', 's'),
    #     ('PUBMED-ORIG', 'GCN', 's'),
    #     ('PUBMED-ORIG', 'FAGCN', 's')
    #     ]:

    # for (dataset_name, model, split_suffix) in [
    #     ('WIKIVITALS_NEW', 'MLP', 's'),
    #     ('WIKIVITALS_NEW', 'GCN', 's'),
    #     ('WIKIVITALS_NEW', 'MLP', '20'),
    #     ('WIKIVITALS_NEW', 'MLP', 'spstrat')
    #     ]:

    for (dataset_name, model, split_suffix) in [
        ('PUBMED-ORIG', 'FAGCN', 's')
        ]:
    


        # ----- CHECK IF CUDA AVAILABLE -----
        torch.cuda.empty_cache()
        cuda = torch.cuda.is_available()

        # ----- LOAD CONFIGURATIONS -----
        opt = json.load(open(f'./configs/grid_search/grid_search_base_{model}.json', 'r', encoding='utf8'))
        opt['dataset'] = dataset_name
    
        # ----- INITIATE GRID SEARCH CLASS -----
        gs = GA_grid_search(opt["grid"])

        # ----- LOAD DATASET -----
        # Loading and pre-processing of data
        X, y_target, adj = load_datasets.load_dataset(opt['dataset'])
        num_nodes, num_features = X.shape
        num_classes = torch.unique(y_target).shape[0]
        opt['num_nodes'], opt['num_features'], opt['num_classes'] = num_nodes, num_features, num_classes
        # Feature binarization (X unchanged if opt['binarize'] == False)
        X = load_datasets.binarize_features(X, opt['binarize'])
        # Feature normalization (X unchanged if opt['normalization'] == "None")
        X = load_datasets.normalize_features(X, opt['normalization'])
        # Binarize y_target 
        y_target_bin = torch.nn.functional.one_hot(y_target)
        # adjacency matrix normalization

        adj_norm = load_datasets.transform_adjacency(
            adj, 
            opt['normalization_trick'],
            opt['to_symmetric'],
            opt['add_self_links']
            )

        

        # ----- GENERATE THE LIST OF ALL INDICES -----
        # Get the list of all indices
        idx_all = torch.LongTensor([i for i in range(opt['num_nodes'])])

        # ----- USE GPUs IF AVAILABLE -----
        if cuda:
            X = X.cuda()
            y_target = y_target.cuda()
            y_target_bin = y_target_bin.cuda()
            # adj = adj.cuda()
            adj_norm = adj_norm.cuda()
            idx_all = idx_all.cuda()

        # ----- INITIALIZE THE LIST OF BEST INDIVIDUALS PER SPLIT -----
        best_individuals = []

        # Let's start to search for best configuration for each split
        for split_idx in range(10):
            gs.reset()
            opt['split_name'] = f'split_{split_idx}_{split_suffix}'

            idx_train, idx_val, idx_test = train_val_test_split(
                splitting_method = opt['splitting_method'],
                dataset = opt['dataset'].lower(),
                split_name = opt['split_name']
            )
            # Update train, val., and test set sizes with their real values 
            opt['train_set_size'] = idx_train.shape[0]
            opt['val_set_size'] = idx_val.shape[0]
            opt['test_set_size'] = idx_test.shape[0]


            # ----- USE GPUs IF AVAILABLE -----
            if cuda:
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()

            # Now that everything is defined, let's redefine the fitness (aka the evaluation 
            # of an individual) which is a function of the various elements previously calculated
            # and which varies from one split to another
            def fitness_GNN_individual(individual, grid_keys):

                # Initialization using the seed provided
                random.seed(opt['seed'])
                torch.manual_seed(opt['seed'])
                if cuda:
                    torch.cuda.manual_seed(opt['seed'])

                current_config = opt
                unstring_list = lambda x: ast.literal_eval(x[4:])
                tmp = tuple([unstring_list(e) if (type(e) == str and e.startswith('LIST')) else e for e in individual])
                individual_config = dict(zip(grid_keys, tmp))

                for k,v in individual_config.items():
                    current_config[k] = v
                
                # ----- INITIALIZE THE MODEL, THE TRAINER, AND SELECT AN EARLY STOPPING STRATEGY -----
                GNN_class  = getattr(sys.modules[__name__], opt['model'])
                if not opt['model'] == 'MLP':
                    GNN = GNN_class(opt, adj_norm)
                else:
                    GNN = GNN_class(opt)
                GNN_trainer = Trainer(opt, GNN)
                if cuda:
                    GNN.cuda()
                # Early_stopping_criterion = early_stopping.LossDecreaseCriterion(patience = opt['early_stopping'])
                Early_stopping_class = getattr(early_stopping, opt['early_stopping_type'])
                Early_stopping_criterion = Early_stopping_class( patience = opt['early_stopping'])


                # ----- TRAIN THE MODEL -----                
                results = []
                max_val_acc = 0.0 # Current maximum validation accuracy
                epoch = 0 # Current epoch
                stop_early = False # Boolean used to decide if early stopping
                # Note: it is not necessary to reset
                GNN_trainer.reset() # model is reset in trainer
                Early_stopping_criterion.reset()

                while not (stop_early or epoch >= opt['epochs']): # 2 stopping criterions: early stopping or max number of epoches reached
                    loss = GNN_trainer.update(X, y_target, idx_train)
                    train_loss, correct_train, preds_train, accuracy_train = GNN_trainer.evaluate(X, y_target, idx_train)
                    val_loss, correct_val, preds_val, accuracy_val = GNN_trainer.evaluate(X, y_target, idx_val)
                    test_loss , correct_test, preds_test, accuracy_test = GNN_trainer.evaluate(X, y_target, idx_test)
                    results += [[epoch, loss, train_loss, accuracy_train, val_loss, accuracy_val, test_loss, accuracy_test]]

                    if accuracy_val >= max_val_acc:
                        # Update maximum validation accuracy
                        max_val_acc = accuracy_val
                        # store the state of the model when validation accuracy is maximum 
                        state = dict([
                            ('model', copy.deepcopy(GNN_trainer.model.state_dict())), 
                            ('optim', copy.deepcopy(GNN_trainer.optimizer.state_dict()))])
                    
                    # Update stop_early to true if the Early stopping criterion is verified
                    stop_early = Early_stopping_criterion.should_stop(epoch, val_loss, accuracy_val)

                    epoch += 1
                return accuracy_val
            
            # Redefine gs fitness
            gs.fitness_individual = lambda x : fitness_GNN_individual(x, gs.grid_keys)


            gs.algo()

            unstring_list = lambda x: ast.literal_eval(x[4:])
            best_individual = [unstring_list(e) if (type(e) == str and e.startswith('LIST')) else e for e in gs.best_individual]
            # best_individual = list(gs.best_individual)
            print(f'{opt["dataset"]} {opt["model"]} {opt["split_name"]}: {best_individual} {gs.max_fitness}')
            best_individuals.append(best_individual)
        
        print(f'RESULTS: {opt["dataset"]} {opt["model"]} {split_suffix}: {best_individuals}')




