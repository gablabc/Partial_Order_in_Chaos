""" Learning algorithms used to train Deep Ensembles """

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import timeit
import wandb

from typing import Optional
from dataclasses import dataclass
from simple_parsing import ArgumentParser


class method_reduce_lr(nn.Module):
    """ Class that trains an Ensemble of Multi-Layered Perceptron by minimizing the training set loss
    and reducing the learning_rate when stagnating for several epochs. """

    @dataclass
    class HParams():
        """
        Args:
            seed (int, optional): Number of the seed used for training. Defaults to 1.
            n_epochs (int, optional): Max. number of epochs to train the ensemble. Defaults to 150.
            n_epochs_log (string, optional): Interval of epochs where valid loss is logged. Default is n_epochs.
            learning_rate (float, optional): Learning rate used in the training of the ensemble. Defaults to 0.001.
            patience (int, optional): Number of stagnating epochs before reducing the learning rate (if use_scheduler is set to True). Defaults to 10.
            lr_decay (float, optional): Decay factor of the learning rate (if use_scheduler is set to True). Defaults to 0.1.
            use_scheduler (bool, optional): To use the reduce_lr_on_plateau scheduler. Defaults to True.
        """
        seed: Optional[int] = 3
        n_epochs: int = 150
        n_epochs_log: str = ""

        def __post_init__(self):  # Hack
            if type(self.n_epochs_log)==str:
                self.n_epochs_log: int = self.n_epochs if self.n_epochs_log == "" else int(self.n_epochs_log)

        learning_rate: float = 0.001 # Learning rate
        patience: int = 10 # Number of epochs before reducing the learning rate (if use_scheduler)
        lr_decay: float = 0.1 # Factor used for lr decay (if use_scheduler)
        use_scheduler: bool = True # To use the reduce_lr_on_plateau scheduler


    def __init__(self, hparams: HParams=None, **kwargs):
        """Initialization of the method_reduce_lr class. The user can either give a premade hparams object 
        made from the Hparams class or give the keyword arguments to make one.

        Args:
            hparams (HParams, optional): HParams object to specify the models characteristics. Defaults to None.
        """
        self.hparams = hparams or self.HParams(**kwargs)
        super().__init__()


    @classmethod
    def from_argparse_args(cls, args):
        """Creates an instance of this Class from the parsed arguments."""
        hparams: cls.HParams = args.method
        return cls(hparams=hparams)


    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        """Adds command-line arguments for this Class to an argument parser."""
        parser.add_arguments(cls.HParams, "method")


    #TODO (small desc)done
    def one_epoch(self):
        """Trains all model of the ensemble of models for one epoch.

        Returns:
            dict: Dictionary that contains the mean loss value (of all (x,y) pairs) and the learning rate at the end of the epoch.
        """
        self.scores = {'loss': 0., }
        example_count = 0.

        self.models.train(True)
        with torch.enable_grad():
            # x being all xs of a batch
            for (x, y) in self.train_loader:
                self.optimizer.zero_grad()

                L = self.models.loss(x, y)
                L.backward()

                lr = self.optimizer.param_groups[0]['lr']

                self.optimizer.step()

                self.scores['loss'] += L.item() * len(x)
                example_count += len(x)
                
                # Log on W&B in real-time
                if self.wandb:
                    wandb.log({'loss': L.item(), 'lr': lr})

        mean_scores = {'loss': self.scores['loss'] / example_count, 'lr': lr}
        
        return mean_scores


    def apply(self, models, train_loader, wandb=False, fresh=True):
        """Method used to train the Ensemble of models and log the loss every n_epochs_log epochs.

        Args:
            model (Ensemble): Ensemble of models that needs to be trained.
            train_loader (Pytorch DataLoader): Loads the mini-batch of training examples.
            wandb (bool, optional): Set to True to log the run on Weights and Biases, False otherwise. Defaults to False.
            fresh (bool, optional): Set to True to start from a fresh set of parameters, False otherwise. Defaults to True.

        Returns:
            float: Time it took to iterate through all the number of epochs where valid loss is logged (n_epochs_log).
        """

    
        # Init additionnal attributes
        self.wandb = wandb
        self.train_loader = train_loader
        self.models = models


        # If starting the procedure
        if fresh:
            # Set optimizer
            self.n_epochs_done = 0
            self.optimizer = torch.optim.Adam(self.models.parameters(),
                                              lr=self.hparams.learning_rate)
            if self.hparams.use_scheduler:
                self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                                   patience=self.hparams.patience, 
                                                   factor=self.hparams.lr_decay, 
                                                   min_lr=0)
            # Main training loop
            # self.logs = {'loss': [], "lr": []}
        start = timeit.default_timer()
        
        with trange(self.n_epochs_done, min(self.hparams.n_epochs,
                                            self.hparams.n_epochs_log + 
                                            self.n_epochs_done)) as tr:
            tr.set_description(desc=self.models.name, refresh=False)
            for _ in tr:
                # if self.n_epochs_done==34:
                #     print("bob")
                scores = self.one_epoch()
                if self.hparams.use_scheduler:
                    self.scheduler.step(scores['loss'])
                    
                # If working in regression report the RSME not the MSE
                if self.models.hparams.task == "regression":
                    # Take RMSE
                    scores['loss'] = scores['loss'] ** (1/2)
                
                # Report
                tr.set_postfix(scores)
                # for key, values in logs.items():
                #     values.append(scores[key])

                #if scores['lr'] <= 1e-4:
                #    break
                self.n_epochs_done += 1
        
        print("\n")
        stop = timeit.default_timer()
        time = stop - start

        return time
