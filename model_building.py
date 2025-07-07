from transformers import AutoTokenizer
from LM import LM_Model
from GNNs import *


def build_LM_model(model_config):
    LM_model = LM_Model(model_config).to(model_config['device'])
    LM_model_name = model_config['lm_model'].lower()
    LM_tokenizer = AutoTokenizer.from_pretrained('...')
    if LM_model_name == 'deberta':
        LM_tokenizer = AutoTokenizer.from_pretrained('...')
    elif LM_model_name == 'roberta-f':
        LM_tokenizer = AutoTokenizer.from_pretrained('...')
    elif LM_model_name == 'roberta':
        LM_tokenizer = AutoTokenizer.from_pretrained('...')
    elif LM_model_name == 'bert':
        LM_tokenizer = AutoTokenizer.from_pretrained('...')
    if LM_model_name != 'roberta-f':
        special_tokens_dict = {'additional_special_tokens': ['DESCRIPTION:','METADATA:','TWEET:']}
        LM_tokenizer.add_special_tokens(special_tokens_dict)
        tokens_list = ["@USER", '#HASHTAG', "HTTPURL", 'EMOJI', 'RT', 'None']
        LM_tokenizer.add_tokens(tokens_list)
        LM_model.LM.resize_token_embeddings(len(LM_tokenizer))
    print('Information about LM model:')
    print('total params:', sum(p.numel() for p in LM_model.parameters()))
    return LM_model, LM_tokenizer


def build_GNN_model(model_config):
    GNN_model_name = model_config['GNN_model'].lower()
    if GNN_model_name == 'sage':
        GNN_model = SAGE(model_config).to(model_config['device'])
    elif GNN_model_name == 'rgcn':
        GNN_model = RGCN(model_config).to(model_config['device'])
    elif GNN_model_name == 'hgt':
        GNN_model = HGT(model_config).to(model_config['device'])
    elif GNN_model_name == 'gat':
        GNN_model = GAT(model_config).to(model_config['device'])
    elif GNN_model_name == 'gcn':
        GNN_model = GAT(model_config).to(model_config['device'])
    else:
        raise ValueError('')

    print('Information about GNN model:')
    print(GNN_model)
    print('total params:', sum(p.numel() for p in GNN_model.parameters()))

    return GNN_model