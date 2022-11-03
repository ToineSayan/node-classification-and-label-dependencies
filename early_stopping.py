import numpy as np


class EarlyStoppingCriterion(object):
    def __init__(self, patience):
        self.patience = patience

    def should_stop(self, epoch, val_loss, val_accuracy):
        raise NotImplementedError

    def after_stopping_ops(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class NoStoppingCriterion(EarlyStoppingCriterion): # not used
    
    def should_stop(self, epoch, val_loss, val_accuracy):
        return False

    def after_stopping_ops(self):
        pass

    def reset(self):
        pass


class GCNCriterion(EarlyStoppingCriterion): # not used
    def __init__(self, patience):
        super().__init__(patience)
        self.val_losses = []

    def should_stop(self, epoch, val_loss, val_accuracy):
        self.val_losses.append(val_loss)

        return epoch >= self.patience and self.val_losses[-1] > np.mean(
            self.val_losses[-(self.patience + 1):-1])

    def after_stopping_ops(self):
        pass

    def reset(self):
        self.val_losses = []


class LossDecreaseCriterion(EarlyStoppingCriterion):
    def __init__(self, patience):
        super().__init__(patience)
        self.min_loss = 100
        self.counter = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_loss < self.min_loss:
            self.counter = 0
            self.min_loss = val_loss
        else:
            self.counter += 1
            
        return self.counter >= self.patience

    def after_stopping_ops(self):
        pass

    def reset(self):
        self.min_loss = 100
        self.counter = 0

class AccuracyAndLossVariationsCriterion(EarlyStoppingCriterion):
    def __init__(self, patience):
        super().__init__(patience)
        self.min_loss = 100
        self.max_accuracy = 0
        self.counter = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_loss < self.min_loss and val_accuracy > self.max_accuracy:
            self.counter = 0
            self.max_accuracy = val_accuracy
            self.min_loss = val_loss
        else:
            self.counter += 1 
            
        return self.counter >= self.patience

    def after_stopping_ops(self):
        pass

    def reset(self):
        self.min_loss = 100
        self.max_accuracy = 0
        self.counter = 0