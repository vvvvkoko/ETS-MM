import json
import torch
from pathlib import Path


def train_val_test_mask():
    train_idx = range(8278)
    val_idx = range(8278, 8278 + 2365)
    test_idx = range(8278 + 2365, 8278 + 2365 + 1183)
    train_idx = torch.tensor(train_idx).to('cuda')
    val_idx = torch.tensor(val_idx).to('cuda')
    test_idx = torch.tensor(test_idx).cpu().numpy()
    return train_idx, val_idx, test_idx


def load_raw_data(dataset, text_path, label_path):
    data_filepath = dataset
    print('Loading data...')
    train_idx, val_idx, test_idx = train_val_test_mask()
    user_text = json.load(open(data_filepath + text_path))  # dtet
    print("usertext_len", len(user_text))
    labels = torch.load(data_filepath + label_path).to(device='cuda')
    return {'train_idx': train_idx,
            'valid_idx': val_idx,
            'test_idx': test_idx,
            'user_text': user_text,
            'labels': labels}


def GNN_load_data(data_filepath):
    train_idx, val_idx, test_idx = train_val_test_mask()
    edge_index = torch.load(data_filepath + 'preprocess/edge_index.pt').to(device='cuda')
    edge_type = torch.load(data_filepath + 'preprocess/edge_type.pt').to(device='cuda')
    num_prop = torch.load(data_filepath + "preprocess/num_prop.pt").to(device='cuda')
    category_prop = torch.load(data_filepath + "preprocess/category_prop.pt").to(device='cuda')
    des_tensor = torch.load(data_filepath + "preprocess/des_tensor.pt").to(device='cuda')
    tweet_tensor1 = torch.load(data_filepath + "preprocess/tweets_tensor_p1.pt").to(device='cuda')
    tweet_tensor2 = torch.load(data_filepath + "preprocess/tweets_tensor_p2.pt").to(device='cuda')
    tweet_tensor = torch.cat([tweet_tensor1, tweet_tensor2], 0)
    return {'train_idx': train_idx,
            'valid_idx': val_idx,
            'test_idx': test_idx,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_prop': num_prop,
            'category_prop': category_prop,
            'des_tensor': des_tensor,
            'tweet_tensor': tweet_tensor
            }


def prepare_path(experiment_name):
    experiment_path = Path(experiment_name)
    ckpt_filepath = experiment_path / 'checkpoints'
    LM_ckpt_filepath = ckpt_filepath / 'LM'
    GNN_ckpt_filepath = ckpt_filepath / 'GNN'
    LM_prt_ckpt_filepath = ckpt_filepath / 'LM_pretrain'
    GNN_prt_ckpt_filepath = ckpt_filepath / 'GNN_pretrain'
    LM_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    LM_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_ckpt_filepath.mkdir(exist_ok=True, parents=True)

    LM_intermediate_data_filepath = experiment_path / 'intermediate' / 'LM'
    GNN_intermediate_data_filepath = experiment_path / 'intermediate' / 'GNN'
    LM_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    GNN_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)

    return LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath, LM_intermediate_data_filepath, GNN_intermediate_data_filepath


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)




