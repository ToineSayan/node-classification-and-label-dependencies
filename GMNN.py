import torch
from trainer import Trainer
import early_stopping
import copy
import sys

from models.gcn import GCN
from models.perceptron import MLP 
from models.graphSAGE import GraphSAGE
from models.fagcn import FAGCN



class GMNN():
    def __init__(self, opt, adj_norm, opt_specific_p = None):
        self.cuda = torch.cuda.is_available()

        self.opt = opt
        self.opt_p = opt_specific_p
        # Initialization using the seed provided
        torch.manual_seed(self.opt['seed'])
        if self.cuda:
            torch.cuda.manual_seed(self.opt['seed'])

        

        self.num_nodes = self.opt['num_nodes']
        self.num_features = self.opt['num_features']
        self.num_classes = self.opt['num_classes']

        ####
        # Init GNNq data and model

        # Initialization of y_q
        self.y_target_base_q = torch.zeros(self.num_nodes, self.num_classes)
        if self.cuda:
            self.y_target_base_q = self.y_target_base_q.cuda()
        

        GNN_class  = getattr(sys.modules[__name__], opt['model'])
        self.model_base_q = GNN_class(opt, adj_norm)
        self.model_base_q.cuda() if self.cuda else self.model_base_q
        self.trainer_base_q = Trainer(opt, self.model_base_q)

        # Early_stopping_criterion = early_stopping.LossDecreaseCriterion(patience = opt['early_stopping'])
        Early_stopping_class = getattr(early_stopping, opt['early_stopping_type'])
        self.Early_stopping_criterion = Early_stopping_class(patience = opt['early_stopping'])


        # ---------------------------------------------------------
        # --------------------- Model p ----------------------
        # ---------------------------------------------------------

        ### Setup of GNNp's input size. 
        # If we decide to use the features of the nodes to train and evaluate GNNp, 
        # then the total size of the input features of GNNp is the number of features
        # of X plus the number of classes
        if self.opt['use_features_p']:
            self.in_size_p = self.num_features + self.num_classes
        else:
            self.in_size_p = self.num_classes

        # X_p_ is the part of the features X_p input to GNNp that varies from one iteration 
        # of GMNN to the next. At each initialization, X_p_ will be updated with the predictions 
        # from GNNq at the previous iteration.
        self.X_p_ = torch.zeros(self.num_nodes, self.num_classes) 
        self.X_p = torch.zeros(self.num_nodes, self.in_size_p)
        
        self.y_target_p = torch.zeros(self.num_nodes, self.num_classes)
        if self.cuda:
            self.X_p = self.X_p.cuda()
            self.X_p_ = self.X_p_.cuda()
            self.y_target_p = self.y_target_p.cuda()

        # Initialization of GNNp
        opt_p = dict()
        for k,v in opt.items(): opt_p[k] = v 
        if not opt_specific_p == None:
            for k,v in opt_specific_p.items(): opt_p[k] = v
        opt_p['num_features'] = self.in_size_p
        GNN_class  = getattr(sys.modules[__name__], opt['model'])
        self.model_p = GNN_class(opt_p, adj_norm)
        self.model_p.cuda() if self.cuda else self.model_p
        self.trainer_p = Trainer(opt_p, self.model_p)

        self.results_to_save = []


    def init_q_data(self, X, y_target, idx_train):
        # self.X_base_q.copy_(X) # The values of X_base_q are the values of X (copy)
        tmp = torch.zeros(idx_train.size(0), self.y_target_base_q.size(1)).type_as(self.y_target_base_q)
        tmp = tmp.cuda() if self.cuda else tmp
        tmp.scatter_(1, torch.unsqueeze(y_target[idx_train], 1), 1.0)
        # y_target_base_q is a size tensor (number of nodes * number of classes). 
        # For each row corresponding to an index of the training set (idx_train), 
        # we find the one-hot representation of the class of the corresponding node. 
        # The other lines are filled with zeros.
        self.y_target_base_q[idx_train] = tmp 


    def update_q_data(self, X, y_target, idx_train):
        # y_target_base_q is updated with the values from the prediction of the previously trained model GNNp
        preds = self.trainer_p.predict(self.X_p, 1.0) # tau == 1 for now
        self.y_target_base_q.copy_(preds)
        # For all the indices of the training set, we use the gold value
        # (we replace these lines by the real classes)
        tmp = torch.zeros(idx_train.size(0), self.y_target_base_q.size(1)).type_as(self.y_target_base_q)
        tmp.scatter_(1, torch.unsqueeze(y_target[idx_train], 1), 1.0)
        self.y_target_base_q[idx_train] = tmp


    def update_p_data(self, X, y_target, idx_train):
        # X_p and y_target_p are initialized with the predictions of the previously trained GNNq model
        preds = self.trainer_base_q.predict(X, self.opt['tau_p']) 
        if self.opt['draw'] == 'exp':
            self.X_p_.copy_(preds)
            self.y_target_p.copy_(preds)
        elif self.opt['draw'] == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            self.X_p_.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            self.y_target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        elif self.opt['draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            self.X_p_.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            self.y_target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)

        # For all the indices of the training set, we use the gold value
        # (we replace these lines by the real classes)
        if self.opt['use_gold_p'] == True:
            tmp = torch.zeros(idx_train.size(0), self.y_target_base_q.size(1)).type_as(self.y_target_base_q)
            tmp.scatter_(1, torch.unsqueeze(y_target[idx_train], 1), 1.0)
            self.X_p_[idx_train] = tmp
            self.y_target_p[idx_train] = tmp

        # Copy X_p_ in X_p 
        self.X_p[:,-self.num_classes:].zero_().copy_(self.X_p_)
        if self.opt['use_features_p'] == True:
            self.X_p[:,:-self.num_classes].zero_().copy_(X)

        

        


    def init_train(self, X, y_target, idx_train, idx_val, idx_test):
        # Initialize data for GNNq
        self.init_q_data(X, y_target, idx_train) 

        results = []
        max_val_acc = 0.0 # Current maximum validation accuracy
        epoch = 0 # Current epoch
        stop_early = False # Boolean used to decide if early stopping
        self.trainer_base_q.reset() # model is reset in trainer
        self.trainer_p.reset()
        self.Early_stopping_criterion.reset()

        while not (stop_early or epoch >= self.opt['epochs']): # 2 stopping criterions: early stopping or max number of epoches reached
            loss = self.trainer_base_q.update_soft(X, self.y_target_base_q, idx_train)
            train_loss, correct_train, preds_train, accuracy_train = self.trainer_base_q.evaluate(X, y_target, idx_train)
            val_loss, correct_val, preds_val, accuracy_val = self.trainer_base_q.evaluate(X, y_target, idx_val)
            test_loss , correct_test, preds_test, accuracy_test = self.trainer_base_q.evaluate(X, y_target, idx_test)
            results += [[epoch, loss, train_loss, accuracy_train, val_loss, accuracy_val, test_loss, accuracy_test]]

            # print(f'Epoch {epoch} - Loss: {loss:.4f}, Validation accuracy: {accuracy_val:.4f}, Test accuracy: {accuracy_test:.4f}')

            if accuracy_val >= max_val_acc:
                # Update maximum validation accuracy
                max_val_acc = accuracy_val
                # store the state of the model when validation accuracy is maximum 
                state = dict([('model', copy.deepcopy(self.trainer_base_q.model.state_dict())), ('optim', copy.deepcopy(self.trainer_base_q.optimizer.state_dict()))])
            
            # Update stop_early to true if the Early stopping criterion is verified
            stop_early = self.Early_stopping_criterion.should_stop(epoch, val_loss, accuracy_val)

            # End of an epoch
            epoch += 1
        
        # The model and the trainer are updated using the state (saved)
        # corresponding to the last maximum validation accuracy 
        self.trainer_base_q.model.load_state_dict(state['model'])
        self.trainer_base_q.optimizer.load_state_dict(state['optim'])

        return results
    


    def train_q(self, X, y_target, idx_train, idx_val, idx_test, idx_all):
        # Update data that will be used to train GNNq
        self.update_q_data(X, y_target, idx_train) # X not used here


        results = []
        epoch = 0 # current epoch
        max_val_acc = 0.0 # Current maximum validation accuracy
        
        while not epoch > self.opt['EM_epochs']: # Only 1 stopping criterion: number of epoch reached
            loss = self.trainer_base_q.update_soft(X, self.y_target_base_q, idx_all)
            train_loss, _, _, accuracy_train = self.trainer_base_q.evaluate(X, y_target, idx_train)
            val_loss, _, _, accuracy_val = self.trainer_base_q.evaluate(X, y_target, idx_val)
            test_loss , _, _, accuracy_test = self.trainer_base_q.evaluate(X, y_target, idx_test)
            # Store the results
            results += [[epoch, loss, train_loss, accuracy_train, val_loss, accuracy_val, test_loss, accuracy_test]]

            if accuracy_val >= max_val_acc:
                # Update maximum validation accuracy
                max_val_acc = accuracy_val
                # store the state of the model when validation accuracy is maximum 
                state = dict([('model', copy.deepcopy(self.trainer_base_q.model.state_dict())), ('optim', copy.deepcopy(self.trainer_base_q.optimizer.state_dict()))])

            # End of an epoch
            epoch += 1

        if self.opt['save_models_during_EM_phases']:
            # The model and the trainer are updated using the state (saved)
            # corresponding to the last maximum validation accuracy 
            self.trainer_base_q.model.load_state_dict(state['model'])
            self.trainer_base_q.optimizer.load_state_dict(state['optim'])
        return results


    def train_p(self, X, y_target, idx_train, idx_val, idx_test, idx_all):
        # Update data that will be used to train GNNp
        self.update_p_data(X, y_target, idx_train)

        results = []
        epoch = 0
        max_val_acc = 0.0 # Current maximum validation accuracy
        # state = dict([('model', copy.deepcopy(self.trainer_p.model.state_dict())), ('optim', copy.deepcopy(self.trainer_p.optimizer.state_dict()))])
        while not epoch > self.opt['EM_epochs']:
            loss = self.trainer_p.update_soft(self.X_p, self.y_target_p, idx_all)
            train_loss, _, _, accuracy_train = self.trainer_p.evaluate(self.X_p, y_target, idx_train)
            val_loss, _, _, accuracy_val = self.trainer_p.evaluate(self.X_p, y_target, idx_val)
            test_loss , _, _, accuracy_test = self.trainer_p.evaluate(self.X_p, y_target, idx_test)
            results += [[epoch, loss, train_loss, accuracy_train, val_loss, accuracy_val, test_loss, accuracy_test]]

            if accuracy_val >= max_val_acc:
                # Update maximum validation accuracy
                max_val_acc = accuracy_val
                # store the state of the model when validation accuracy is maximum 
                state = dict([('model', copy.deepcopy(self.trainer_p.model.state_dict())), ('optim', copy.deepcopy(self.trainer_p.optimizer.state_dict()))])
            # End of an epoch
            epoch += 1
        
        if self.opt['save_models_during_EM_phases']:
            # The model and the trainer are updated using the state (saved)
            # corresponding to the last maximum validation accuracy
            self.trainer_p.model.load_state_dict(state['model'])
            self.trainer_p.optimizer.load_state_dict(state['optim'])

        return results


    def reset(self):
        # INFO: trainer.reset() resets the model & the optimizer
        self.trainer_base_q.reset() 
        self.trainer_p.reset()
    
    def pretrain_q(self, X, y_target, idx_train, idx_val, idx_test):
        r = self.init_train(X, y_target, idx_train, idx_val, idx_test)
        self.results_to_save.append(r)
        best_val_acc, test_acc = max([(r_e[5], r_e[7]) for r_e in r])
        # print(f'Initial training ~ results:\tbest validation accuracy:{best_val_acc}, test accuracy: {test_acc}')
        return {'val':best_val_acc, 'test': test_acc}
    
    def do_EM_phases(self, X, y_target, idx_train, idx_val, idx_test, idx_all):
        # GMNN loops
        if self.opt['GMNN_architecture']:
            rq, rp = [], []
            for i in range(self.opt['EM_loops']):
                r = self.train_p(X, y_target, idx_train, idx_val, idx_test, idx_all)
                # self.results_to_save.append(r)
                rp.append(r)
                best_val_acc_p, test_acc_p, loop = max([(r_epoch[5], r_epoch[7], i+1) for i in range(len(rp)) for r_epoch in rp[i]])

                r = self.train_q(X, y_target, idx_train, idx_val, idx_test, idx_all)
                rq.append(r)
                best_val_acc_q, test_acc_q, loop = max([(r_epoch[5], r_epoch[7], i+1) for i in range(len(rq)) for r_epoch in rq[i]])
                # print(f'EM loop {i+1}:\t val accuracy GNNp: {best_val_acc_p}, val accuracy GNNq: {best_val_acc_q}')
                # print(f'EM loop {i+1}:\t test accuracy GNNp: {test_acc_p}, test accuracy GNNq: {test_acc_q}')
                val_p_last_loop, test_p_last_loop = max([(r_epoch[5], r_epoch[7]) for r_epoch in rp[-1]])
                val_q_last_loop, test_q_last_loop = max([(r_epoch[5], r_epoch[7]) for r_epoch in rq[-1]])
            return {
                'val_q':best_val_acc_q, 'val_p': best_val_acc_p, 'test_p': test_acc_p, 'test_q': test_acc_q,
                'val_q_last_loop': val_q_last_loop, 'val_p_last_loop': val_p_last_loop,
                'test_q_last_loop': test_q_last_loop, 'test_p_last_loop': test_p_last_loop
            }
        else:
            return {}
    
    def do_one_train_p(self, X, y_target, idx_train, idx_val, idx_test, idx_all):
        # GMNN loops
        if self.opt['GMNN_architecture']:
            rp = []
            r = self.train_p(X, y_target, idx_train, idx_val, idx_test, idx_all)
            # self.results_to_save.append(r)
            rp.append(r)
            best_val_acc_p, test_acc_p, loop = max([(r_epoch[5], r_epoch[7], i+1) for i in range(len(rp)) for r_epoch in rp[i]])
        return {'val_q': 0, 'val_p': best_val_acc_p, 'test_p': test_acc_p, 'test_q': 0}







