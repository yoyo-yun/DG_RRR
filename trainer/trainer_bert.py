import os
import time
import torch
import datetime
import numpy as np

from tqdm import tqdm
from models import *
from transformers import BertTokenizer
from trainer.utils import multi_acc, multi_mse, load_datasetbert_from_local, multi_f1_macro, multi_f1_micro
from models.get_optim import get_Adam_optim_v2


ALL_MODLES = {
    'bert': PSBERT.BertForSequenceClassification,
}


class Trainer:
    def __init__(self, config):
        self.config = config
        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.train_itr, self.dev_itr, self.test_itr, self.usr_stoi, self.prd_stoi = load_datasetbert_from_local(config)
        self.config.moniter_per_step = len(self.train_itr) // 10

        # print(config)
        net = ALL_MODLES[config.model].from_pretrained(pretrained_weights,
                                                       num_labels=config.num_labels,
                                                       cus_config=config,
                                                       return_dict=False)
        net.bert.init_personalized()

        self.KD = KD_zoo.BiSelfKD(T1=4, T2=4)

        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
            self.optim, self.scheduler = get_Adam_optim_v2(config, self.net.module, lr_rate=config.lr_base)
        else:
            self.net = net.to(self.config.device)
            self.optim, self.scheduler = get_Adam_optim_v2(config, self.net, lr_rate=config.lr_base)

        self.early_stop = config.early_stop
        self.best_dev_acc = 0
        self.eval_f1_macro = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.step_count = 0
        self.oom_time = 0
        self.losses = []
        self.losses_whole = []
        self.dev_acc = []

        training_steps_per_epoch = len(self.train_itr) // (config.gradient_accumulation_steps)
        self.config.num_train_optimization_steps = self.config.max_epoch * training_steps_per_epoch

        self.log_file = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt'


    def get_logging(self, loss, acc, rmse, f1_mi=None, f1_ma=None, eval='training'):
        logs_metrics_format = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            "total_loss:{:>2.4f}\ttotal_acc:{:>2.4f}\ttotal_rmse:{:>2.4f}"
        logs_f1_format = "\ttotal_f1_macro:{:>2.4f}\ttotal_f1_micro:{:>2.4f}\t"

        if f1_mi is not None and f1_mi is not None:
            logs_format = logs_metrics_format + logs_f1_format
            logs = logs_format.format(loss, acc, rmse, f1_ma, f1_mi)  + "\n"
        else:
            logs_format = logs_metrics_format
            logs = logs_format.format(loss, acc, rmse) + "\n"
        return logs

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def eval(self, eval_itr, agnostic=False):
        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_label = []
        total_logit = []
        eval_oom = 0
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            input_ids, label, usr, prd = batch
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != 0).long()
            labels = label.long().to(self.config.device)
            usr = [self.usr_stoi[x] for x in usr]
            prd = [self.prd_stoi[x] for x in prd]
            usr = torch.Tensor(usr).long().to(self.config.device)
            prd = torch.Tensor(prd).long().to(self.config.device)
            try:
                logits = self.net(input_ids=input_ids,
                                  user_ids=usr,
                                  item_ids=prd,
                                  attention_mask=attention_mask,
                                  agnostic=agnostic)[0]
                loss = loss_fn(logits, labels)

                total_loss.append(loss.item())
                total_label.extend(label.cpu().detach().tolist())
                total_logit.extend(logits.cpu().detach().tolist())

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        eval_oom += 1
                else:
                    torch.cuda.empty_cache()
                    print(str(exception))
                    raise exception

            # monitoring results on every steps1
            end_time = time.time()
            span_time = (end_time - start_time) * (int(len(eval_itr)) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            if eval_oom > 0:
                print("out of memory times over evalation is:", eval_oom)
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr)),
                    100 * (step) / int(len(eval_itr)),
                    int(h), int(m), int(s)),
                end="")

        return np.array(total_loss).mean(0), \
               metric_fn(torch.tensor(total_label), torch.tensor(total_logit)), \
               mse_fn(torch.tensor(total_label), torch.tensor(total_logit)).sqrt(), \
               multi_f1_micro(torch.tensor(total_label), torch.tensor(total_logit)), \
               multi_f1_macro(torch.tensor(total_label), torch.tensor(total_logit))

    def save_state(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.ensureDirs(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        self.tokenizer.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        if self.config.n_gpu > 1:
            self.net.module.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        else:
            self.net.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))

    def load_state(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        net = ALL_MODLES[self.config.model].from_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset), cus_config=self.config)
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
        else:
            self.net = net.to(self.config.device)

    def ensureDirs(self, *dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def run(self, run_mode):
        if run_mode == 'train':
            # empty log file
            self.empty_log()
            self.train()
            self.load_state()
            # do test
            self.net.eval()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro, "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)

                eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)

        if run_mode == 'test':
            self.load_state()
            # do test
            self.net.eval()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr, agnostic=False)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro, "testing-enhanced")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)

                eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro = self.eval(self.test_itr, agnostic=True)
                eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_macro, eval_f1_micro,
                                             "testing-agnostic")
                print("\r" + eval_logs)
                self.logging(
                    self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                    eval_logs)
        else:
            exit(-1)

    def empty_log(self):
        if (os.path.exists(self.log_file)):
            os.remove(self.log_file)
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def test_load_dataset(self):
        for step, batch in enumerate(self.train_itr):
            self.step_count += 1
            self.net.train()
            input_ids, label, usr, prd = batch
            # input_ids = processor4baseline(text, self.tokenizer, self.config)

            usr = torch.Tensor([self.usr_stoi[x] for x in usr]).long().to(self.config.device)
            prd = torch.Tensor([self.prd_stoi[x] for x in prd]).long().to(self.config.device)

            print(input_ids)
            print(label)
            print(usr)
            print(prd)

    def train(self):
        # Save log information
        logfile = open(self.log_file, 'a+')
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n' +
            'data:' + str(self.config.dataset) +
            '\n' +
            'strategy:' + str(self.config.strategy) +
            '\n'
        )
        logfile.close()
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        eval_best_loss = 0.
        eval_best_acc = 0.
        eval_best_rmse = 0.
        eval_best_f1_micro = 0.
        eval_best_f1_macro = 0.
        self.optim.zero_grad()
        for epoch in range(0, self.config.max_epoch):
            epoch_tqdm = tqdm(self.train_itr)
            epoch_tqdm.set_description_str("Processing Epoch: {}".format(epoch))
            for step, batch in enumerate(epoch_tqdm):
                self.step_count += 1
                self.net.train()
                input_ids, label, usr, prd = batch
                input_ids = input_ids.to(self.config.device)
                attention_mask = (input_ids != 0).long() # id of [PAD] is 0
                labels = label.long().to(self.config.device)
                usr = torch.Tensor([self.usr_stoi[x] for x in usr]).long().to(self.config.device)
                prd = torch.Tensor([self.prd_stoi[x] for x in prd]).long().to(self.config.device)
                try:
                    logits1 = self.net(input_ids=input_ids,
                                      user_ids=usr,
                                      item_ids=prd,
                                      attention_mask=attention_mask,
                                      agnostic=False,
                                      )[0]
                    logits2 = self.net(input_ids=input_ids,
                                      user_ids=usr,
                                      item_ids=prd,
                                      attention_mask=attention_mask,
                                      agnostic=True,
                                      )[0]
                    kd_loss = self.KD(logits1, logits2)
                    loss = loss_fn(logits1, labels) + kd_loss
                    metric_acc = acc_fn(labels, logits1)
                    metric_mse = mse_fn(labels, logits1)
                    total_loss.append(loss.item())
                    total_acc.append(metric_acc.item())
                    total_mse.append(metric_mse.item())

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                    loss.backward()

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.scheduler.step()
                        self.optim.zero_grad()

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        self.oom_time += 1
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                        print(str(exception))
                        raise exception

                if self.step_count % self.config.moniter_per_step == 0:
                    # evaluating phase
                    self.net.eval()
                    with torch.no_grad():
                        eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.dev_itr, agnostic=False)
                        self.dev_acc.append(eval_acc)

                    # monitoring eval metrices
                    if eval_f1_macro > eval_best_f1_macro:
                        eval_best_loss = eval_loss
                        eval_best_acc = eval_acc
                        eval_best_rmse = eval_rmse
                        eval_best_f1_micro = eval_f1_micro
                        eval_best_f1_macro = eval_f1_macro

                    if eval_f1_macro > self.eval_f1_macro:
                        # saving models
                        self.eval_f1_macro = eval_f1_macro
                        self.save_state()
                        self.unimproved_iters = 0
                    else:
                        self.unimproved_iters += 1
                        if self.unimproved_iters >= self.config.patience and self.early_stop == True:
                            early_stop_logs = self.log_file + "\n" + \
                                              "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch,
                                                                                                   self.best_dev_acc)
                            print(early_stop_logs)
                            self.logging(self.log_file, early_stop_logs)

                            # load best model on dev datasets
                            self.load_state()
                            # logging test logs
                            self.net.eval()
                            with torch.no_grad():
                                eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro = self.eval(self.test_itr)

                            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval_f1_micro, eval_f1_macro,
                                                         eval="testing")
                            print("\r" + eval_logs)
                            # logging testt logs
                            self.logging(self.log_file, eval_logs)
                            return
                            # exit()
            # monitoring stats at each epoch
            train_loss, train_acc, train_rmse = \
                np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())

            logs = ("    Epoch:{:^5}    ".format(epoch)).center(85, "-") \
                   + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)
            if self.oom_time > 0:
                print("num of out of memory is: " + str(self.oom_time))
            # logging training logs
            self.logging(self.log_file, logs)

            eval_logs = self.get_logging(eval_best_loss,
                                         eval_best_acc,
                                         eval_best_rmse,
                                         eval_best_f1_micro,
                                         eval_best_f1_macro,
                                         eval="evaluating")
            print("\r" + eval_logs)
            # logging evaluating logs
            self.logging(self.log_file, eval_logs)

            # reset monitors
            total_loss = []
            total_acc = []
            total_mse = []
            eval_best_loss = 0.
            eval_best_acc = 0.
            eval_best_rmse = 0.
            eval_best_f1_micro = 0.
            eval_best_f1_macro = 0.
