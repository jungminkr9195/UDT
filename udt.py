# The script is based on the code of Anomaly-Transformer (https://github.com/thuml/Anomaly-Transformer)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, lr, patience=7, verbose=False, dataset_name='', seed=0, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.seed = seed
        self.lr = lr

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Udt(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = os.path.join(self.model_save_path, f'seed_{self.seed}')
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(lr=self.lr, patience=3, verbose=True, dataset_name=self.dataset, seed=self.seed)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, _) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
    
    def udt(self):
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), f"seed_{self.seed}", str(self.dataset) + '_checkpoint.pth')))
        temperature = self.temperature
        since = time.time()
        print("======================Dynamic UDT======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        train_energy, _ = self.get_anomaly_score(self.train_loader, self.model, criterion, temperature=temperature)

        # (2) find the fixed threshold
        thre_energy, _ = self.get_anomaly_score(self.thre_loader, self.model, criterion, temperature=temperature)
        combined_energy = np.concatenate([train_energy, thre_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold: ", thresh)

        # (3) Uncertainty Quantification
        test_energy, test_labels = self.get_anomaly_score(self.test_loader, self.model, criterion, temperature=temperature)
        print('-'*10, 'Uncertainty Quantification', '-'*10)
        data_unc_tr, model_unc_tr = self.get_uncertainty(train_energy, self.train_loader, self.model)
        data_unc_t, model_unc_t = self.get_uncertainty(train_energy, self.test_loader, self.model)
        
        data_unc = np.array(list(data_unc_tr) + list(data_unc_t))
        model_unc = np.array(list(model_unc_tr) + list(model_unc_t))
        beta2 = data_unc.mean()/model_unc.mean()
        
        total_unc_v = (1-self.beta)*data_unc_tr + beta2*self.beta*model_unc_tr
        uncertainty_t = (1-self.beta)*data_unc_t + beta2*self.beta*model_unc_t

        mean_uncertainty_v = np.mean(total_unc_v)
        std_uncertainty_v = np.std(total_unc_v)
        
        # (4) Dynamic Thresholding
        thr_list = self.get_threshold(thresh, mean_uncertainty_v, std_uncertainty_v, uncertainty_t)
        preds_dynamic = []
        preds_fix = (test_energy > thresh).astype(int)
        for i in range(len(test_energy)):
            if test_energy[i] > thr_list[i]:
                preds_dynamic.append(1)
            else:
                preds_dynamic.append(0)
                
        preds_dynamic = np.array(preds_dynamic)
        preds_dynamic = preds_dynamic.astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", preds_dynamic.shape)
        print("gt:     ", gt.shape)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and preds_dynamic[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if preds_dynamic[j] == 0:
                            preds_dynamic[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if preds_dynamic[j] == 0:
                            preds_dynamic[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                preds_dynamic[i] = 1

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and preds_fix[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if preds_fix[j] == 0:
                            preds_fix[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if preds_fix[j] == 0:
                            preds_fix[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                preds_fix[i] = 1
                
        pred = np.array(preds_dynamic)
        pred_fix = np.array(preds_fix)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')
        
        accuracy_fix = accuracy_score(gt, pred_fix)
        precision_fix, recall_fix, f_score_fix, _ = precision_recall_fscore_support(gt, pred_fix, average='binary')
        
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        print("Accuracy_fix : {:0.4f}, Precision_fix : {:0.4f}, Recall_fix : {:0.4f}, F-score_fix : {:0.4f} ".format(accuracy_fix, precision_fix, recall_fix, f_score_fix))
        
        required_time = time.time() - since
        print(f'Time: {round(required_time, 3)}')
        print('-' * 10, f'Data {self.dataset} seed {self.seed} a {self.a} beta {self.beta} mcdo {self.mcdo} c {self.c} finished', '-' * 10)
        
        return accuracy, precision, recall, f_score
    
    
    def get_anomaly_score(self, loader, model, criterion, temperature):
        temperature = self.temperature
        attens_energy, total_labels = [], []
        model.eval()
        for i, (input_data, labels) in enumerate(tqdm(loader)):
            input = input_data.float().to(self.device)
            output, series, prior, _ = model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            total_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        total_labels = np.concatenate(total_labels, axis=0).reshape(-1)
        attens_energy = np.array(attens_energy)
        total_labels = np.array(total_labels)
        
        return attens_energy, total_labels 
    
    def cal_score(self, batch, model, temperature):
        temperature = self.temperature
        input = batch.float().to(self.device)
        output, series, prior, _ = model(input)
        criterion = nn.MSELoss(reduce=False)
        loss = torch.mean(criterion(input, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss

        return cri
    
    def get_uncertainty(self, train_error, loader, model):
        
        with torch.no_grad():
            data_unc_lst, model_unc_lst = [], []
            train_error = torch.Tensor(train_error).to(self.device)

            for idx, (batch, _) in enumerate(tqdm(loader)):
                batch_data = batch.float().to(self.device)

                scores = [self.cal_score(batch_data, model, temperature=self.temperature) for i in range(self.mcdo)]
                mc_loss = torch.stack(scores)
                mc_loss = mc_loss.reshape(self.mcdo, -1)

                for i in range(mc_loss.shape[0]):
                    for j in range(mc_loss.shape[1]):
                        mc_loss[i, j] = (mc_loss[i, j] >= train_error).sum() / len(train_error)

                mc_loss1 = mc_loss.mul(1-mc_loss)
                data_unc = torch.mean(mc_loss1, dim=0).detach().cpu().numpy()
                
                model_unc = torch.var(mc_loss, dim=0, unbiased=False, keepdim=True).detach().cpu().numpy()
                model_unc = model_unc.reshape(-1)

                data_unc_lst.append(data_unc)
                model_unc_lst.append(model_unc)

            data_unc_lst = np.concatenate(data_unc_lst, axis=0)
            model_unc_lst = np.concatenate(model_unc_lst, axis=0)

        return data_unc_lst, model_unc_lst 
    
    def get_threshold(self, init_thr, valid_mean_uncertainty, valid_std_uncertainty, test_uncertainty):
        thr_list = []
        for i in range(len(test_uncertainty)):
            uncertainty = (test_uncertainty[i] - (1 + self.c) * valid_mean_uncertainty) / valid_std_uncertainty
            threshold = init_thr - self.a * uncertainty
            thr_list.append(threshold)

        return thr_list
    
