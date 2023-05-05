import os
import time
import torch
import datetime
import numpy as np

from tqdm import tqdm
from models.get_optim import get_Adam_optim, get_Adam_optim_v2
from trainer.utils import multi_acc, multi_mse, load_vectors_LSTM, load_vectors, multi_f1_macro, multi_f1_micro
from models import *
# import models

ALL_MODLES = {
    'lstm': BiLSTM.Net,
    'nsc': HAN.Net,
    'cnn': CNN.Net
}


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.step_count = 0
        self.losses = []
        self.losses_whole = []
        self.dev_acc_per_epoch = []
        self.best_dev_acc = 0

    def ensureDirs(self, *dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def train(self):
        pass

    def train_epoch(self):
        pass

    def eval(self, eval_itr):
        pass

    def empty_log(self, version):
        if (os.path.exists(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')):
            os.remove(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, loss, acc, rmse, f1_mac=None, f1_mic=None, eval='training'):
        logs_metrics_format = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            "total_loss:{:>2.4f}\ttotal_acc:{:>2.4f}\ttotal_rmse:{:>2.4f}"
        logs_f1_format = "\ttotal_f1_macro:{:>2.4f}\ttotal_f1_micro:{:>2.4f}\t"

        if f1_mic is not None and f1_mac is not None:
            logs_format = logs_metrics_format + logs_f1_format
            logs = logs_format.format(loss, acc, rmse, f1_mac, f1_mic)  + "\n"
        else:
            logs_format = logs_metrics_format
            logs = logs_format.format(loss, acc, rmse) + "\n"
        return logs


class BaselineTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.train_itr, self.dev_itr, self.test_itr = load_vectors_LSTM(config)

        net = ALL_MODLES[config.model](self.config).to(self.config.device)
        self.KD = KD_zoo.BiSelfKD(T1=4, T2=4)
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net)
        else:
            self.net = net
        self.optim = get_Adam_optim(config, self.net)

        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.best_dev_mac_f1 = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.moniter_per_step = len(self.train_itr) // 10

    def train(self):
        # Save log information
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.close()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.net.train()
            train_loss, train_acc, train_rmse, \
            best_losses_per_epoch, best_acc_per_epoch, best_rmse_per_peoch, best_mic_f1_per_peoch, best_mac_f1_per_peoch = \
                self.train_epoch(agnostic=False)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         logs)

            eval_logs = self.get_logging(best_losses_per_epoch,
                                         best_acc_per_epoch,
                                         best_rmse_per_peoch,
                                         best_mac_f1_per_peoch,
                                         best_mic_f1_per_peoch,
                                         eval="evaluating")
            print("\r" + eval_logs)

            # early stopping
            if best_mac_f1_per_peoch > self.best_dev_mac_f1:
                self.unimproved_iters = 0
                self.best_dev_mac_f1 = best_mac_f1_per_peoch
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                    early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt' + "\n" + \
                                      "Early Stopping. Epoch: {}, Best Dev Macro-F1: {}".format(epoch, self.best_dev_mac_f1)
                    print(early_stop_logs)
                    self.logging(
                        self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                        early_stop_logs)
                    break

    def train_epoch(self, epoch=1, agnostic=False):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        eval_best_loss = 0.
        eval_best_acc = 0.
        eval_best_rmse = 0.
        eval_best_mac_f1 = 0.
        eval_best_mic_f1 = 0.
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {}".format(epoch))
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            self.optim.zero_grad()
            text, sent_length = batch.text
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd
            logits1 = self.net(text, usr, prd, mask=(text != self.config.pad_idx), agnostic=False)
            logits2 = self.net(text, usr, prd, mask=(text != self.config.pad_idx), agnostic=True)
            kd_loss = self.KD(logits1, logits2)
            loss = loss_fn(logits1, label) + kd_loss
            metric_acc = acc_fn(label, logits1)
            metric_mse = mse_fn(label, logits1)
            loss.backward()
            self.optim.step()

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())
            total_mse.append(metric_mse.item())

            if step % self.moniter_per_step == 0 and step != 0:
                self.net.eval()
                with torch.no_grad():
                    eval_loss, eval_acc, eval_rmse, eval_mic_f1, eval_mac_f1 = self.eval(self.dev_itr, agnostic=False)

                # monitoring eval metrices
                if eval_mac_f1 > eval_best_mac_f1:
                    eval_best_loss = eval_loss
                    eval_best_acc = eval_acc
                    eval_best_rmse = eval_rmse
                    eval_best_mic_f1 = eval_mic_f1
                    eval_best_mac_f1 = eval_mac_f1

                if eval_mac_f1 > self.best_dev_mac_f1:
                    self.saving_model()

        return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean()), \
               eval_best_loss, eval_best_acc, eval_best_rmse, eval_best_mic_f1, eval_best_mac_f1

    def saving_model(self):
        state = {
            'state_dict': self.net.module.state_dict() if self.config.n_gpu > 1 else self.net.state_dict(),
            'usr_vectors': {
                'stoi': self.train_itr.dataset.USR_FIELD.vocab.stoi,
                'itos': self.train_itr.dataset.USR_FIELD.vocab.itos,
            },
            'prd_vectors': {
                'stoi': self.train_itr.dataset.PRD_FIELD.vocab.stoi,
                'itos': self.train_itr.dataset.PRD_FIELD.vocab.itos,
            },
            'text_vectors': {
                'stoi': self.train_itr.dataset.TEXT_FIELD.vocab.stoi,
                'itos': self.train_itr.dataset.TEXT_FIELD.vocab.itos,
            }
        }
        torch.save(
            state,
            self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
        )

    def loading_model(self):
        state = torch.load(self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version))['state_dict']
        self.net.load_state_dict(state)

    def eval(self, eval_itr, agnostic=False):
        self.net.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_label = []
        total_logit = []
        self.net.eval()
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            text, sent_length = batch.text
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd
            logits = self.net(text, usr, prd, mask=(text != self.config.pad_idx), agnostic=agnostic)
            loss = loss_fn(logits, label)
            total_loss.append(loss.item())
            total_label.extend(label.cpu().detach().tolist())
            total_logit.extend(logits.cpu().detach().tolist())

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(eval_itr.dataset) / self.config.TRAIN.batch_size) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    int(h), int(m), int(s)),
                end="")

        return np.array(total_loss).mean(0), \
               metric_fn(torch.tensor(total_label), torch.tensor(total_logit)), \
               mse_fn(torch.tensor(total_label), torch.tensor(total_logit)).sqrt(), \
               multi_f1_micro(torch.tensor(total_label), torch.tensor(total_logit)), \
               multi_f1_macro(torch.tensor(total_label), torch.tensor(total_logit))


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
            self.loading_model()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        elif run_mode == 'val':
            self.loading_model()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.dev_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        elif run_mode == 'test':
            self.loading_model()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)

                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        else:
            exit(-1)


class HieTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.train_itr, self.dev_itr, self.test_itr = load_vectors(config)

        net = ALL_MODLES[config.model](self.config).to(self.config.device)
        self.KD = KD_zoo.BiSelfKD(T1=4, T2=4)

        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net)
        else:
            self.net = net
        self.optim = get_Adam_optim(config, self.net)

        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.best_dev_mac_f1 = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.moniter_per_step = len(self.train_itr) // 10


    def train(self):
        # Save log information
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.close()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.net.train()
            train_loss, train_acc, train_rmse,\
            best_losses_per_epoch, best_acc_per_epoch, best_rmse_per_peoch, best_mic_f1_per_peoch, best_mac_f1_per_peoch =\
                self.train_epoch(agnostic=False)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         logs)

            eval_logs = self.get_logging(best_losses_per_epoch,
                                         best_acc_per_epoch,
                                         best_rmse_per_peoch,
                                         best_mac_f1_per_peoch,
                                         best_mic_f1_per_peoch,
                                         eval="evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         eval_logs)

            # early stopping
            if best_mac_f1_per_peoch > self.best_dev_mac_f1:
                self.unimproved_iters = 0
                self.best_dev_mac_f1 = best_mac_f1_per_peoch
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                    early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt' + "\n" + \
                                      "Early Stopping. Epoch: {}, Best Dev Macro-F1: {}".format(epoch, self.best_dev_mac_f1)
                    print(early_stop_logs)
                    self.logging(
                        self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                        early_stop_logs)
                    break

    def train_epoch(self, epoch=1, agnostic=False):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        eval_best_loss = 0.
        eval_best_acc = 0.
        eval_best_rmse = 0.
        eval_best_mac_f1 = 0.
        eval_best_mic_f1 = 0.
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {}".format(epoch))
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            self.optim.zero_grad()
            text, doc_length, sent_length = batch.text
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd
            logits1, doc_repr_1, (usr_en, prd_en) = self.net(text, usr, prd, mask=(text != self.config.pad_idx), agnostic=False)
            logits2, doc_repr_2, (usr_ag, prd_ag) = self.net(text, usr, prd, mask=(text != self.config.pad_idx), agnostic=True)
            kd_loss = self.KD(logits1, logits2)
            loss = loss_fn(logits1, label) + kd_loss
            metric_acc = acc_fn(label, logits1)
            metric_mse = mse_fn(label, logits1)
            loss.backward()
            self.optim.step()

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())
            total_mse.append(metric_mse.item())


            if step % self.moniter_per_step == 0 and step != 0:
                self.net.eval()
                with torch.no_grad():
                    eval_loss, eval_acc, eval_rmse, eval_mic_f1, eval_mac_f1 = self.eval(self.dev_itr, agnostic=False)

                # monitoring eval metrices
                if eval_mac_f1 > eval_best_mac_f1:
                    eval_best_loss = eval_loss
                    eval_best_acc = eval_acc
                    eval_best_rmse = eval_rmse
                    eval_best_mic_f1 = eval_mic_f1
                    eval_best_mac_f1 = eval_mac_f1

                if eval_mac_f1 > self.best_dev_mac_f1:
                    # saving models
                    self.saving_model()

        return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean()),\
               eval_best_loss, eval_best_acc, eval_best_rmse, eval_best_mic_f1, eval_best_mac_f1

    def saving_model(self):
        # saving state
        state = {
            'state_dict': self.net.state_dict(),
            'usr_vectors': {
                'stoi': self.train_itr.dataset.USR_FIELD.vocab.stoi,
                'itos': self.train_itr.dataset.USR_FIELD.vocab.itos,
                # 'dim': self.train_itr.dataset.USR_FIELD.vocab.dim
            },
            'prd_vectors': {
                'stoi': self.train_itr.dataset.PRD_FIELD.vocab.stoi,
                'itos': self.train_itr.dataset.PRD_FIELD.vocab.itos,
                # 'dim': self.train_itr.dataset.PRD_FIELD.vocab.dim
            },
            'text_vectors': {
                'stoi': self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.stoi,
                'itos': self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.itos,
                # 'dim': self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.dim
            }
        }
        torch.save(
            state,
            self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
        )

    def loading_model(self):
        state = torch.load(self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version))['state_dict']
        self.net.load_state_dict(state)


    def eval(self, eval_itr, agnostic=False):
        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_label = []
        total_logit = []
        self.net.eval()
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            text, doc_length, sent_length = batch.text
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd

            logits, doc_repr, _ = self.net(text, usr, prd, mask=(text != self.config.pad_idx), agnostic=agnostic)
            loss = loss_fn(logits, label)
            total_loss.append(loss.item())
            total_label.extend(label.cpu().detach().tolist())
            total_logit.extend(logits.cpu().detach().tolist())

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(eval_itr.dataset) / self.config.TRAIN.batch_size) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    int(h), int(m), int(s)),
                end="")

        return np.array(total_loss).mean(0), \
               metric_fn(torch.tensor(total_label), torch.tensor(total_logit)), \
               mse_fn(torch.tensor(total_label), torch.tensor(total_logit)).sqrt(), \
               multi_f1_micro(torch.tensor(total_label), torch.tensor(total_logit)), \
               multi_f1_macro(torch.tensor(total_label), torch.tensor(total_logit))


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
            self.loading_model()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro, "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro, "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        elif run_mode == 'val':
            self.loading_model()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.dev_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,  "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro, "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        elif run_mode == 'test':
            self.loading_model()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro, "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)

                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro, "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        else:
            exit(-1)
