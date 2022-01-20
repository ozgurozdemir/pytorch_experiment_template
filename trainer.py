from sklearn.metrics import confusion_matrix

import datetime, time, os

import numpy as np
import copy

import torch
from torch import nn


class Trainer():
    def __init__(self, 
                 model_func, model_args, 
                 optim, optim_args,
                 loss_func, 
                 device, 
                 epochs, metrics, verbose,
                 model_checkpoint_path, experiment_save_path):
        self.model_func = model_func
        self.model_args = model_args
        self.loss_func  = loss_func
        self.optim      = optim
        self.optim_args = optim_args
        self.device     = device
        self.epochs     = epochs
        self.verbose    = verbose
        self.metrics    = metrics
        
        self.model_checkpoint_path = model_checkpoint_path
        self.experiment_save_path  = experiment_save_path
    
    
    # ========================================================================
    #                            EXPERIMENT FUNCTIONS
    # ========================================================================
    def make_cross_validation_experiment_one_out(self, dataset, folds, experiment_note):      
        # file corresponding to the experiment
        file_name = datetime.datetime.now().strftime("%c").replace(" ", "_").replace(":", "-")
        
        for fold in folds:
            # preparing the dataset
            train_ds, valid_ds, test_ds = dataset.prepare_speaker_independent_sets(fold["train_set"],
                                                                                   fold["valid_set"],
                                                                                   fold["test_set"])
            # initializing the model
            model = self.model_func(**self.model_args).to(self.device)
            optim = self.optim(model.parameters(), **self.optim_args)
            loss_func = self.loss_func()
            
            # training
            best_model = self.train_model(model, loss_func, optim, train_ds, valid_ds)
            
            # testing
            preds, gts, results = self._eval_model(best_model, test_ds, return_pred=True)
            print(self._log_experiment_results(fold, preds, gts, results))
            
            # saving the experiment
            self._save_experiment(file_name, experiment_note, fold, preds, gts, results)
            
    
    def make_single_shot_experiment(self, train_dataset, valid_dataset, test_dataset):
        
        # initializing the model
        model = self.model_func(**self.model_args).to(self.device)
        optim = self.optim(model.parameters(), **self.optim_args)
        loss_func = self.loss_func()
        
        # training
        best_model = self.train_model(model, loss_func, optim, train_dataset, valid_dataset)
        
        # test
        preds, gts, results = self._eval_model(best_model, test_dataset, return_pred=True)
        print(self._log_experiment_results("custom", preds, gts, results))
        
        return best_model
        
    
    
    # ========================================================================
    #                              SAVE/LOAD MODEL
    # ========================================================================
    def save_model(self, model):
        pass
    
    def load_model(self, load_path):
        pass
    
    
    
    # ========================================================================
    #                            TRAINING FUNCTIONS
    # ========================================================================
    def train_model(self, 
                    model, loss_func, optim, 
                    train_dataset, valid_dataset, 
                    save_best_metric="UA", save_after=10):        
        best_metric = -np.inf
        
        for epoch in range(self.epochs):
            if self.verbose: 
                print(f">> Epoch {epoch+1} started...")
            
            # initialize variables for training
            start_time = time.time()
            training_loss = 0
                        
            # training
            model.train()
            for batch, (inp, tar) in enumerate(train_dataset):
                training_loss += self._train_step(model, loss_func, optim, inp, tar)
            
            training_loss /= (batch+1)
            
            # evaluation
            train_metrics = self._eval_model(model, train_dataset, return_pred=False)
            valid_metrics = self._eval_model(model, valid_dataset, return_pred=False)
            
            # information about epoch
            if self.verbose: 
                self._print_epoch_info(epoch+1, training_loss, train_metrics, valid_metrics,
                                       save_best_metric, best_metric, start_time)
            
            # copy the best model
            if epoch > save_after and valid_metrics[save_best_metric] > best_metric:
                best_metric = valid_metrics[save_best_metric]
                best_model = copy.deepcopy(model)
        
        return best_model
    
    
    def _train_step(self, model, loss_func, optim, inp, tar):
        inp = inp.to(self.device)
        tar = torch.argmax(tar, dim=1).long()
        tar = tar.to(self.device)
        
        # feed-forward
        out  = model(inp)
        loss = loss_func(out, tar)
        
        # back-propagation
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        return loss.data
    
    
    def _eval_model(self, model, dataset, return_pred):
        pred = []; gt = []
        
        # making the predictions
        model.eval()
        
        with torch.no_grad():    
            for batch, (inp, tar) in enumerate(dataset):
                inp = inp.to(self.device)
                out = model(inp).detach().cpu().numpy()
                
                pred.append(out)
                gt.append(tar.numpy())
        
        # calculating the metrics
        (pred, gt) = (np.concatenate(pred), np.concatenate(gt))
        results = {m: self.metrics[m](pred, gt) for m in self.metrics}
        
        if return_pred:
            return pred, gt, results
        else:
            return results
    
    # ========================================================================
    #                              LOG FUNCTIONS
    # ========================================================================
    def _print_epoch_info(self, epoch, 
                          training_loss, train_metric,
                          valid_metric, save_metric, best_result,
                          epoch_start_time):
        print(f":: Epoch {epoch} is comleted...")
        print(f":: \t\033[92mTraining loss: {training_loss}") 
        print(f":: \t\033[92mTraining metrics: {train_metric}")
        print(f":: \t\033[91mValidation metrics: {valid_metric}")
        print(f":: \t\033[1mBest {save_metric} metric: {best_result}")
        print(f":: \t\033[0mEpoch is completed in {time.time() - epoch_start_time} sec...\n")
    
    
    def _log_test_results(self, predictions, groundtruths, metric_results):
        cnfs_matrix = confusion_matrix(groundtruths.argmax(axis=1), predictions.argmax(axis=1))
        
        return (
            "=" * 45 + 
            f"\nConfusion mat:\n{str(cnfs_matrix)}\n" + "-"*25 +
            f"\nMetric Results: {metric_results}\n" +
            "=" * 45 + "\n\n"
        )
    
    
    def _log_experiment_results(self, set_dist, predictions, groundtruths, metric_results):
        return (
            f"Experiment is completed at {datetime.datetime.now().strftime('%c')}\n\n" +
            f"Model: {self.model_func}, Args: {self.model_args}\n" +
            f"Set distribution: {set_dist}\n" +
            self._log_test_results(predictions, groundtruths, metric_results)
        )
    
    
    def _save_experiment(self, file_name, experiment_note, set_dist, pred, gt, metric_res):
        file_path = f"{self.experiment_save_path}/{file_name}.txt"
        
        # if there is continuing experiment 
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                exp_info = file.read()
        else:
            exp_info = f"Experiment Note: {experiment_note}\n\n"
        
        # write the experiment information
        exp_info += self._log_experiment_results(set_dist, pred, gt, metric_res)
        
        with open(file_path, "w") as file:
            file.write(exp_info)