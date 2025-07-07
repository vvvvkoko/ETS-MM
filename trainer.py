from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from model_building import build_LM_model, build_GNN_model
from dataloader import build_LM_dataloader
import os
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.models import MLP
from utils import *
import torch.nn as nn


class LM_Trainer:
    def __init__(
            self,
            model_name,
            classifier_n_layers,
            classifier_hidden_dim,
            device,
            pretrain_epochs,
            optimizer_name,
            lr,
            weight_decay,
            dropout,
            att_dropout,
            lm_dropout,
            activation,
            warmup,
            label_smoothing_factor,
            pl_weight,
            max_length,
            batch_size,
            grad_accumulation,
            lm_epochs_per_iter,
            temperature,
            pl_ratio,
            eval_patience,
            intermediate_data_filepath,
            ckpt_filepath,
            pretrain_ckpt_filepath,
            raw_data_filepath,
            train_idx,
            valid_idx,
            test_idx,
            hard_labels,
            user_seq,
    ):

        self.model_name = model_name
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lm_dropout = lm_dropout
        self.warmup = warmup
        self.label_smoothing_factor = label_smoothing_factor
        self.pl_weight = pl_weight
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.lm_epochs_per_iter = lm_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.eval_patience = eval_patience
        self.intermediate_data_filepath = intermediate_data_filepath
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.raw_data_filepath = Path(raw_data_filepath)
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.user_seq = user_seq
        self.do_mlm_task = False

        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_epoch = 0
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')
        self.results = {}

        self.get_train_idx_all()
        self.pretrain_steps_per_epoch = self.train_idx.shape[0] // self.batch_size + 1
        self.pretrain_steps = int(self.pretrain_steps_per_epoch * self.pretrain_epochs)
        self.train_steps_per_iter = (self.train_idx_all.shape[0] // self.batch_size + 1) * self.lm_epochs_per_iter
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'lm_model': model_name,
            'dropout': dropout,
            'att_dropout': att_dropout,
            'lm_dropout': self.lm_dropout,
            'classifier_n_layers': classifier_n_layers,
            'classifier_hidden_dim': classifier_hidden_dim,
            'activation': activation,
            'device': device,
            'return_mlm_loss': True if self.do_mlm_task else False
        }

        self.dataloader_config = {
            'batch_size': batch_size,
            'pl_ratio': pl_ratio
        }

    def build_model(self):
        self.model, self.tokenizer = build_LM_model(self.model_config)
        self.DESCRIPTION_id = self.tokenizer.convert_tokens_to_ids('[Description]')
        self.TWEET_id = self.tokenizer.convert_tokens_to_ids('[Tweet]')
        self.TOPIC_id = self.tokenizer.convert_tokens_to_ids('[Topic]')
        self.EMOTION_id = self.tokenizer.convert_tokens_to_ids('[Emotion]')
        # self.METADATA_id = self.tokenizer.convert_tokens_to_ids('METADATA:')

    def get_optimizer(self, parameters):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(parameters, **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(parameters, **self.optimizer_args)
        else:
            return NotImplementedError
        return optimizer

    def get_scheduler(self, optimizer, mode='train'):
        if mode == 'pretrain':
            return get_cosine_schedule_with_warmup(optimizer, self.pretrain_steps_per_epoch * self.warmup,
                                                   self.pretrain_steps)
        else:
            return CosineAnnealingLR(optimizer, T_max=self.train_steps_per_iter, eta_min=0)


    def train_infer(self, user_seq, hard_labels):
        self.model.eval()
        embeddings = []
        all_outputs = []
        all_labels = []
        infer_loader = build_LM_dataloader(self.dataloader_config, None, user_seq, hard_labels,
                                           mode='infer')
        with torch.no_grad():
            ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            for batch in tqdm(infer_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                embedding, output = self.model(tokenized_tensors)
                embeddings.append(embedding.cpu())
                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            test_embeddings = torch.cat(embeddings)
            print(test_embeddings.shape)
            torch.save(test_embeddings, self.intermediate_data_filepath / 'embeddings_iter.pt')
            test_acc = accuracy(all_outputs[self.test_idx], all_labels[self.test_idx])
            output = all_outputs.max(1)[1].to('cpu').detach().numpy()
            label = all_labels.to('cpu').detach().numpy()
            test_f1 = f1_score(output[self.test_idx], label[self.test_idx])
        print(f'LM train Test Accuracy = {test_acc}')
        print(f'LM train Test F1 = {test_f1}')

    def train(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.get_optimizer(parameters)
        scheduler = self.get_scheduler(optimizer)

        print('LM training start!')
        step = 0
        train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx_all, self.user_seq, self.hard_labels,
                                           'train', self.is_pl)
        for epoch in range(self.lm_epochs_per_iter):
            self.model.train()
            print(f'This is iter {self.iter} epoch {epoch}/{self.lm_epochs_per_iter - 1}')

            for batch in tqdm(train_loader):
                step += 1
                tokenized_tensors, labels, is_pl = self.batch_to_tensor(batch)
                _, output = self.model(tokenized_tensors)

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()
                labels1 = F.one_hot(labels).float()

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels1[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels1[pl_idx])
                else:
                    pl_tensor = F.log_softmax(output[pl_idx] / self.temperature, dim=-1)
                    loss_KD = self.KD_criterion(pl_tensor, labels1[pl_idx].float())
                    loss_H = self.criterion(output[rl_idx], labels1[rl_idx])
                    loss = self.pl_weight * loss_KD + (1 - self.pl_weight) * loss_H
                loss /= self.grad_accumulation
                loss.backward()

                if step % self.grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                if step % self.eval_patience == 0:
                    valid_acc, valid_f1 = self.eval()
                    print(f'LM Valid Accuracy = {valid_acc}')
                    print(f'LM Valid F1 = {valid_f1}')
                    if valid_acc > self.best_valid_acc:
                        self.best_valid_acc = valid_acc
                        self.best_epoch = epoch
                        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return

    def eval(self, mode='valid'):
        if mode == 'valid':
            eval_loader = build_LM_dataloader(self.dataloader_config, self.valid_idx, self.user_seq, self.hard_labels,
                                              mode='eval')
        elif mode == 'test':
            eval_loader = build_LM_dataloader(self.dataloader_config, self.test_idx, self.user_seq, self.hard_labels,
                                              mode='eval')
        self.model.eval()
        valid_predictions = []
        valid_labels = []
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                _, output = self.model(tokenized_tensors)
                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(labels.cpu().numpy())
            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions)
            return valid_acc, valid_f1

    def test(self):
        print('Computing test accuracy and f1 for LM...')
        ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval('test')
        print(f'LM Test Accuracy = {test_acc}')
        print(f'LM Test F1 = {test_f1}')
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def batch_to_tensor(self, batch):
        if len(batch) == 3:
            tokenized_tensors = self.tokenizer(text=batch[0], return_tensors='pt', max_length=self.max_length,
                                               truncation=True, padding='longest', add_special_tokens=False)
            for key in tokenized_tensors.keys():
                tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
            labels = batch[1].to(self.device)
            is_pl = torch.tensor(batch[2]).to(self.device)
            return tokenized_tensors, labels, is_pl
        elif len(batch) == 2:
            tokenized_tensors = self.tokenizer(text=batch[0], return_tensors='pt', max_length=self.max_length,
                                               truncation=True, padding='longest', add_special_tokens=False)
            for key in tokenized_tensors.keys():
                tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
            labels = batch[1].to(self.device)
            return tokenized_tensors, labels, None

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)

    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.cpu().numpy())
        n = self.train_idx.shape[0]
        pl_ratio_LM = min(self.pl_ratio, (n_total - n) / n)
        n_pl_LM = int(n * pl_ratio_LM)
        pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_LM, replace=False)).to('cuda')
        self.train_idx_all = torch.cat((self.train_idx, pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0


class GNN_Trainer:
    def __init__(self,
                 model_name,
                 device,
                 optimizer_name,
                 lr,
                 weight_decay,
                 dropout,
                 pl_weight,
                 batch_size,
                 gnn_n_layers,
                 n_relations,
                 activation,
                 gnn_epochs_per_iter,
                 temperature,
                 pl_ratio,
                 intermediate_data_filepath,
                 ckpt_filepath,
                 pretrain_ckpt_filepath,
                 train_idx,
                 valid_idx,
                 test_idx,
                 hard_labels,
                 edge_index,
                 edge_type,
                 num_prop,
                 category_prop,
                 des_tensor,
                 tweet_tensor,
                 SimpleHGN_att_res,
                 att_heads,
                 RGT_semantic_heads,
                 gnn_hidden_dim,
                 lm_name
                 ):
        self.model_name = model_name
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.pl_weight = pl_weight
        self.dropout = dropout
        self.batch_size = batch_size
        self.gnn_n_layers = gnn_n_layers
        self.n_relations = n_relations
        self.activation = activation
        self.gnn_epochs_per_iter = gnn_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.intermediate_data_filepath = intermediate_data_filepath
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_prop = num_prop,
        self.category_prop = category_prop,
        self.des_tensor = des_tensor
        self.tweet_tensor = tweet_tensor
        self.SimpleHGN_att_res = SimpleHGN_att_res
        self.att_heads = att_heads
        self.RGT_semantic_heads = RGT_semantic_heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lm_input_dim = 1024 if lm_name.lower() in ['roberta-large'] else 768
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')
        self.loss = nn.CrossEntropyLoss()

        self.results = {}
        self.get_train_idx_all()
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'GNN_model': model_name,
            'optimizer': optimizer_name,
            'gnn_n_layers': gnn_n_layers,
            'n_relations': n_relations,
            'activation': activation,
            'dropout': dropout,
            'gnn_hidden_dim': gnn_hidden_dim,
            'lm_input_dim': self.lm_input_dim,
            'SimpleHGN_att_res': SimpleHGN_att_res,
            'att_heads': att_heads,
            'RGT_semantic_heads': RGT_semantic_heads,
            'device': device
        }

        self.dataloader_config = {
            'batch_size': batch_size,
            'n_layers': gnn_n_layers
        }

    def build_model(self):
        self.model = build_GNN_model(self.model_config)

    def get_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.gnn_epochs_per_iter, eta_min=0)

    def get_optimizer(self):

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            return NotImplementedError

        return optimizer

    def train(self, embeddings_LM, all_labels):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        label = all_labels.to(self.device)
        print('GNN training start!')
        for epoch in tqdm(range(self.gnn_epochs_per_iter)):
            self.model.train()
            output, em = self.model(embeddings_LM.to(self.device),
                                    self.edge_index, self. edge_type, self.num_prop[0], self.category_prop[0],
                                    self.des_tensor, self.tweet_tensor)
            loss_train = self.loss(output[self.train_idx], label[self.train_idx])
            acc_train = accuracy(output[self.train_idx], label[self.train_idx])
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            print('train:',
                  'Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train))

            valid_acc, valid_f1 = self.eval(embeddings_LM, all_labels)
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}, self.ckpt_filepath / 'best.pkl')

        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return None

    def eval(self, embeddings_LM, all_labels):
        self.model.eval()
        label = all_labels.to(self.device)
        with torch.no_grad():
            output,em = self.model(embeddings_LM.to(self.device),
                                   self.edge_index, self. edge_type, self.num_prop[0], self.category_prop[0],
                                   self.des_tensor, self.tweet_tensor)
            valid_acc = accuracy(output[self.valid_idx], label[self.valid_idx])
            output = output.max(1)[1].cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            valid_idx = self.valid_idx.cpu().numpy()
            valid_f1 = f1_score(output[valid_idx], label[valid_idx])
        return valid_acc, valid_f1

    def test(self, embeddings_LM, all_labels):
        print('Computing test accuracy and f1 for GNN...')
        ckpt = torch.load(self.ckpt_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        label = all_labels.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output, em = self.model(embeddings_LM.to(self.device),
                                    self.edge_index, self. edge_type, self.num_prop[0], self.category_prop[0],
                                    self.des_tensor, self.tweet_tensor)
            test_acc = accuracy(output[self.test_idx], label[self.test_idx])
            output = output.max(1)[1].to('cpu').detach().numpy()
            label = label.to('cpu').detach().numpy()
            test_f1 = f1_score(output[self.test_idx], label[self.test_idx])
        print(f'GNN Test Accuracy = {test_acc}')
        print(f'GNN Test F1 = {test_f1}')
        self.results['accuracy'] = test_acc.item()
        self.results['f1'] = test_f1
        return None

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)

    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.cpu().numpy())
        n = self.train_idx.shape[0]
        pl_ratio_GNN = min(self.pl_ratio, (n_total - n) / n)
        n_pl_GNN = int(n * pl_ratio_GNN)
        self.pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_GNN, replace=False)).to('cuda')
        self.train_idx_all = torch.cat((self.train_idx, self.pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0
