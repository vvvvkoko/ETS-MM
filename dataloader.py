from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data


class LM_dataset(Dataset):
    def __init__(self, user_text: list, labels: torch.Tensor, is_pl: torch.LongTensor=None):
        super().__init__()
        self.user_text = user_text
        self.labels = labels
        self.is_pl = is_pl
        
    def __getitem__(self, index):
        if self.is_pl is None:
            text = self.user_text[index]
            label = self.labels[index]
            return text, label
        else:
            text = self.user_text[index]
            label = self.labels[index]
            is_pl = self.is_pl[index]
            return text, label, is_pl

    def __len__(self):
        return len(self.user_text)


def build_LM_dataloader(dataloader_config, idx, user_seq, labels, mode, is_pl=None):
    batch_size = dataloader_config['batch_size']

    print('train')
    print(mode)

    if mode == 'train':
        user_text = []
        for i in idx:
            user_text.append(user_seq[i.item()])
        loader = DataLoader(dataset=LM_dataset(user_text, labels[idx], is_pl), batch_size=batch_size, shuffle=True)

    elif mode == 'pretrain':
        user_text = []
        print(len(idx))
        print(len(user_seq))
        for i in idx:
            user_text.append(user_seq[i.item()])
        print("pretrain", len(user_text))
        dataset = LM_dataset(user_text, labels[idx], is_pl)
        # dataset = LM_dataset(user_text, labels[idx])
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)  # 返回了NoneType, 可能数据中存在一些None值？？

    elif mode == 'infer':
        loader = DataLoader(dataset=LM_dataset(user_seq, labels), batch_size=batch_size*5)

    elif mode == 'eval':
        user_text = []
        for i in idx:
            user_text.append(user_seq[i.item()])
        print("eval", len(user_text))
        loader = DataLoader(dataset=LM_dataset(user_text, labels[idx], is_pl), batch_size=batch_size*5)
    
    else:
        raise ValueError('mode should be in ["train", "eval", "infer", "pretrain"].')

    return loader


def build_GNN_dataloader(dataloader_config, idx, LM_embedding, labels, edge_index, edge_type,
                         num_prop, num_category, mode, is_pl=None):
    batch_size = dataloader_config['batch_size']
    n_layers = dataloader_config['n_layers']
    data = Data(x=LM_embedding, edge_index=edge_index, edge_type=edge_type, labels=labels)
    data.num_nodes = LM_embedding.shape[0]
    print("numnodes", data.num_nodes)
    data.num_prop = num_prop
    data.num_category = num_category
    if mode == 'train' or mode == 'pretrain':
        data.is_pl = is_pl
        loader = NeighborLoader(data=data, num_neighbors=[-1] * n_layers, batch_size=batch_size,
                                shuffle=True)
    elif mode == 'eval':
        loader = NeighborLoader(data=data, num_neighbors=[-1]*n_layers, batch_size=batch_size)
    elif mode == 'infer':
        loader = NeighborLoader(data=data, num_neighbors=[-1]*n_layers, batch_size=batch_size)
    else:
        raise ValueError('mode should be in ["train", "valid", "test", "infer"].')

    return loader


