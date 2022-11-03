from torch import nn
import torch
from torch.cuda.amp import GradScaler, autocast


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

class Trainer(object):
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        if opt['decay_policy'] == [-1]:
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        else:
            to_apply = lambda n: any([n.startswith(f'layers.{i}') for i in opt['decay_policy']])
            self.param0 = [p for n, p in self.model.named_parameters() if p.requires_grad and to_apply(n)]
            self.param1 = [p for n, p in self.model.named_parameters() if p.requires_grad and not to_apply(n)]
            self.parameters = [{'params': self.param0}, {'params': self.param1, 'weight_decay':0}]

        if torch.cuda.is_available():
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset_parameters()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
        self.model.train()
        self.optimizer.zero_grad() #we need to set the gradients to zero before starting to do backpropagation because PyTorch accumulates the gradients on subsequent backward passes
        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, target_binarized, idx):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target_binarized[idx]* logits[idx], dim=-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def evaluate(self, inputs, target, idx):
        self.model.eval()
        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)
        return loss.item(), correct, preds, accuracy.item()

    def predict(self, inputs, tau=1):
        self.model.eval()
        logits = self.model(inputs) / tau
        logits = torch.softmax(logits, dim=-1).detach()
        return logits

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])











class GMNN_Trainer(object):
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.trainer_q = Trainer(opt, model.GNNq)
        self.trainer_p = Trainer(opt, model.GNNp)

    def reset(self):
        self.model.reset_parameters()
        self.trainer_p.reset_parameters()
        self.trainer_q.reset_parameters()

    def update(self, inputs, target, idx):
        self.model.train()
        self.optimizer.zero_grad() #we need to set the gradients to zero before starting to do backpropagation because PyTorch accumulates the gradients on subsequent backward passes
        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, target_binarized, idx):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target_binarized[idx]* logits[idx], dim=-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def evaluate(self, inputs, target, idx):
        self.model.eval()
        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)
        return loss.item(), correct, preds, accuracy.item()

    def predict(self, inputs, tau=1):
        self.model.eval()
        logits = self.model(inputs) / tau
        logits = torch.softmax(logits, dim=-1).detach()
        return logits

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])