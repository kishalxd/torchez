from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.something = 'Hi'
        self.train_loader = None
        self.valid_loader = None
        self.predict_loader = None
        self.train_dataset = None
        self.valid_dataset = None
        self.pred_dataset = None
        self.train_batch_size = None
        self.valid_batch_size = None
        self.epochs = None
        self.device = None
        self.es = None
        self.es_monitor = 'train_loss'
        self.model_state = 'train'
        self.epoch_monitor = None
        self.model_path = None
        self.metric = None
        self.es_counter = 0
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    def loss_fn(self, *args, **kwargs):
        return

    def training_step(self, data):

        outputs = self(data)
        loss = self.loss_fn(outputs, targets)
        metric = self.metrics(outputs, targets)
        return {'loss':loss, 'metric':metric}

    def validation_step(self, data):

        outputs = self(data)
        loss = self.loss_fn(outputs, targets)

        return {'loss':loss}

    def prediction_step(self, data):

        outputs = self(data)

        return outputs
        
    def train_one_epoch(self, epoch, optimizer):
        self.train()
        assert self.train_loader is not None, 'train loader not initialised'
        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=True)
        bar.set_description('Train')
        for idx, data in bar:
            for key, value in data.items():
                data[key] = value.to(self.device)
            loss = self.training_step(**data)
            
            step_loss = loss['loss']
            
            step_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            loss['train_loss'] = step_loss.item()
            del loss['loss']

            if len(loss)>1 :
                metric = loss.copy()
                del metric['train_loss']
                bar.set_postfix(Epoch=epoch, loss=step_loss.item(), **metric)
            else:
                bar.set_postfix(Epoch=epoch, loss=step_loss.item())
        return loss
            
    def valid_one_epoch(self, epoch):
        self.eval()
        assert self.valid_loader is not None, 'train loader not initialised'
        epoch_loss = 0
        epoch_metric = 0
        dataset_size = 0
        total_loss = 0
        total_metric = 0

        with torch.no_grad():
            bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), dynamic_ncols=True)
            bar.set_description('Valid')

            for idx, data in bar:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                loss = self.validation_step(**data)

                # step_loss = loss['loss']
                # loss['valid_loss'] = step_loss.item()
                # del loss['loss']
                for key, val in data.items():
                    batch_size = val.shape[0]
                    break
                
                step_loss = loss['loss'].item()
                running_loss = step_loss * batch_size
                dataset_size+= batch_size
                total_loss = total_loss + running_loss
                epoch_loss = total_loss / dataset_size

                if len(loss)>1 :
                    metric = loss.copy()
                    del metric['loss']

                    metrics_to_show={}
                    for key,value in metric.items():
                        running_metric = value * batch_size
                        total_metric = total_metric + running_metric
                        epoch_metric = total_metric / dataset_size
                        metrics_to_show[key] = epoch_metric

                    bar.set_postfix(Epoch=epoch, loss=epoch_loss, **metrics_to_show)
                else:
                    bar.set_postfix(Epoch=epoch, loss=epoch_loss)

                to_return = {}
                for key, value in metrics_to_show.items():
                    to_return[f'valid_{key}'] = value
                to_return['valid_loss'] = epoch_loss

        return to_return
    
    def get_predictions(self):
        self.eval()
        assert self.pred_loader is not None, 'train loader not initialised'
        all_outputs=[]
        with torch.no_grad():
            bar = tqdm(enumerate(self.pred_loader), total=len(self.pred_loader))
            bar.set_description('Inference')

            for idx, data in bar:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                outputs = self.prediction_step(**data)
                outputs = outputs.cpu().detach().numpy()
                all_outputs.extend(outputs)
        return all_outputs
    
    def metrics(self, outputs, targets):
        return 
        
    def get_optimizer(self):
        return 
    
    def fit(
        self, 
        train_dataset, 
        train_batch_size=None,
        valid_dataset=None,
        valid_batch_size=None,
        epochs=10,
        device='cpu',
        es=False,
        es_epochs=1,
        es_mode='min',
        es_monitor='valid_loss',
        model_path=None,
        save=False
    ):
        
        self.device = device
        self.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_loader = DataLoader(self.train_dataset, self.train_batch_size)
        self.valid_loader = DataLoader(self.valid_dataset, self.valid_batch_size)
        self.epochs = epochs
        self.optimizer = self.get_optimizer()
        self.model_path = model_path

        self.es = es
        self.es_epochs = es_epochs
        self.es_mode = es_mode
        self.es_monitor = es_monitor

        if self.es_mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

        if 'train_' in self.es_monitor:
            self.model_state = 'train'
        elif 'valid_' in self.es_monitor:
            self.model_state = 'valid'

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch, self.optimizer)
            self.train_epoch_end(train_loss)
            
            if self.valid_dataset is not None:
                val_loss = self.valid_one_epoch(epoch)
                stop_or_not = self.valid_epoch_end(val_loss)
                if stop_or_not == 'stop':
                    break

        if self.es:
            print(f'Best {self.es_monitor} : {self.best_score}')

    def train_epoch_end(self, train_loss):
        if self.model_state == 'train':
            if self.es:
                self.epoch_monitor = train_loss[self.es_monitor]
                if self.es_mode == 'min':
                     

                    if self.epoch_monitor < self.best_score:                    
                        print(f'{self.es_monitor} improved : {self.best_score} --------------> {self.epoch_monitor}')
                        self.best_score = self.epoch_monitor
                        self.save()
                        print('Model saved')

                    else:
                        print('Do es')
                else:

                    if self.epoch_monitor > self.best_score:                    
                        print(f'validation {self.es_monitor} improved : {self.best_score} --------------> {self.epoch_monitor}')
                        self.best_score = self.epoch_monitor
                        self.save()
                        print('Model saved')

                    else:
                        print('Do es')

    def valid_epoch_end(self, val_loss):
        if self.model_state == 'valid':

            if self.es:
                self.epoch_monitor = val_loss[self.es_monitor]
                print(val_loss)
                if self.es_mode == 'min':
                        

                    if self.epoch_monitor < self.best_score:  
                        self.es_counter = 0                  
                        print(f'{self.es_monitor} improved : {self.best_score} --------------> {self.epoch_monitor}')
                        self.best_score = self.epoch_monitor
                        self.save()
                        print('Model saved')

                    else:
                        self.es_counter+=1
                        print(f'Early Stopping Counter {self.es_counter} of {self.es_epochs}')

                        if self.es_counter == self.es_epochs:
                            print('******* Early Stopping *******')
                            return 'stop'
                else:

                    if self.epoch_monitor > self.best_score:                    
                        print(f'validation {self.es_monitor} improved : {self.best_score} --------------> {self.epoch_monitor}')
                        self.best_score = self.epoch_monitor
                        self.save()
                        print('Model saved')

                    else:
                        self.es_counter+=1
                        print(f'Early Stopping Counter {self.es_counter} of {self.es_epochs}')

                        if self.es_counter == self.es_epochs:
                            print('******* Early Stopping *******')
                            return 'stop'

            else:
                self.save()
                print('Model saved')

    def save(self):
        model_state_dict = self.state_dict()
        torch.save(model_state_dict, self.model_path)
    
    def predict(self, pred_dataset, batch_size=1, device='cpu', model_path=None):
        self.device = device
        self.to(self.device)
        if model_path is not None:
            model_dict = torch.load(model_path, map_location=torch.device(device))
            self.load_state_dict(model_dict)
        self.pred_dataset = pred_dataset
        self.pred_loader = DataLoader(self.pred_dataset, batch_size)
        preds = self.get_predictions()
        return preds