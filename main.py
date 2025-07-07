from trainer import LM_Trainer, GNN_Trainer
from utils import *
from parser_args import parser_args
import os


def main(args):
    (LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath,
     LM_intermediate_data_filepath, GNN_intermediate_data_filepath) = prepare_path(args.experiment_name)
    # load data for LM
    data = load_raw_data(dataset='...',  # setting dataset path
                         text_path='...',  # setting text path
                         label_path=''  # setting label path
                         )

    LMTrainer = LM_Trainer(
        model_name=args.LM_model,
        classifier_n_layers=args.LM_classifier_n_layers,
        classifier_hidden_dim=args.LM_classifier_hidden_dim,
        device=args.device,
        pretrain_epochs=args.LM_pretrain_epochs,
        optimizer_name=args.optimizer_LM,
        lr=args.lr_LM,
        weight_decay=args.weight_decay_LM,
        dropout=args.LM_dropout,
        att_dropout=args.LM_att_dropout,
        lm_dropout=args.LM_dropout,
        warmup=args.warmup,
        label_smoothing_factor=args.label_smoothing_factor,
        pl_weight=args.alpha,
        max_length=args.max_length,
        batch_size=args.batch_size_LM,
        grad_accumulation=args.LM_accumulation,
        lm_epochs_per_iter=args.LM_epochs_per_iter,
        temperature=args.temperature,
        pl_ratio=args.pl_ratio_LM,
        intermediate_data_filepath=LM_intermediate_data_filepath,
        ckpt_filepath=LM_ckpt_filepath,
        pretrain_ckpt_filepath=LM_prt_ckpt_filepath,
        raw_data_filepath=args.raw_data_filepath,
        train_idx=data['train_idx'],
        valid_idx=data['valid_idx'],
        test_idx=data['test_idx'],
        hard_labels=data['labels'],
        user_seq=data['user_text'],
        eval_patience=args.LM_eval_patience,
        activation=args.activation
    )

    LMTrainer.build_model()

    if not os.path.exists(LM_intermediate_data_filepath / 'embeddings_iter.pt'):
        if not os.listdir(LM_ckpt_filepath) or not os.path.exists(
                args.experiment_name + '/results_LM.json'):
            LMTrainer.train()
            LMTrainer.save_results(args.experiment_name + f'/results_LM.json')
        # load data for LM
        data = load_raw_data(dataset='...',
                             text_path='...',
                             label_path='...')
        LMTrainer.train_infer(data['user_text'], data['labels'])


    # load data for GNN
    data = GNN_load_data(data_filepath='...')
    embeddings_LM1 = torch.load(LM_intermediate_data_filepath / f'embeddings_iter.pt')

    if embeddings_LM1.shape != data['tweet_tensor'].shape:
        pad = data['tweet_tensor'].size(0) - embeddings_LM1.size(0)
        padding_tensor = torch.zeros(pad, 768)
        embeddings_LM = torch.cat((embeddings_LM1, padding_tensor), dim=0)
    else:
        embeddings_LM = embeddings_LM1
    all_labels_pre_LM = torch.load('/root/Twibot20/preprocess/labels.pt')

    # GNN model
    GNNTrainer = GNN_Trainer(
        model_name=args.GNN_model,
        device=args.device,
        optimizer_name=args.optimizer_GNN,
        lr=args.lr_GNN,
        weight_decay=args.weight_decay_GNN,
        dropout=args.GNN_dropout,
        pl_weight=args.beta,
        batch_size=args.batch_size_GNN,
        gnn_n_layers=args.n_layers,
        n_relations=args.n_relations,
        activation=args.activation,
        gnn_epochs_per_iter=args.GNN_epochs_per_iter,
        temperature=args.temperature,
        pl_ratio=args.pl_ratio_GNN,
        intermediate_data_filepath=GNN_intermediate_data_filepath,
        ckpt_filepath=GNN_ckpt_filepath,
        pretrain_ckpt_filepath=GNN_prt_ckpt_filepath,
        train_idx=data['train_idx'],
        valid_idx=data['valid_idx'],
        test_idx=data['test_idx'],
        hard_labels=all_labels_pre_LM,
        edge_index=data['edge_index'],
        edge_type=data['edge_type'],
        num_prop=data['num_prop'],
        category_prop=data['category_prop'],
        des_tensor=data['des_tensor'],
        tweet_tensor=data['tweet_tensor'],
        SimpleHGN_att_res=args.SimpleHGN_att_res,
        att_heads=args.att_heads,
        RGT_semantic_heads=args.RGT_semantic_heads,
        gnn_hidden_dim=args.hidden_dim,
        lm_name=args.LM_model
    )
    GNNTrainer.build_model()
    GNNTrainer.train(embeddings_LM, all_labels_pre_LM)
    GNNTrainer.test(embeddings_LM, all_labels_pre_LM)
    GNNTrainer.save_results(args.experiment_name + f'/results_GNN.json')


def GNNTrain():
    LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath, LM_intermediate_data_filepath, GNN_intermediate_data_filepath = prepare_path(
        args.experiment_name)
    data = GNN_load_data(data_filepath='/root/autodl-tmp/data/Twibot20/')
    all_labels_pre_LM = torch.load('/root/autodl-tmp/data/Twibot20/preprocess/labels.pt')
    embeddings_LM1 = torch.load(LM_intermediate_data_filepath / f'embeddings_iter.pt')
    if embeddings_LM1.shape != data['tweet_tensor'].shape:
        pad = data['tweet_tensor'].size(0) - embeddings_LM1.size(0)
        padding_tensor = torch.zeros(pad, 768)
        embeddings_LM = torch.cat((embeddings_LM1, padding_tensor), dim=0)
    else:
        embeddings_LM = embeddings_LM1

    # GNN model
    GNNTrainer = GNN_Trainer(
        model_name=args.GNN_model,
        device=args.device,
        optimizer_name=args.optimizer_GNN,
        lr=args.lr_GNN,
        weight_decay=args.weight_decay_GNN,
        dropout=args.GNN_dropout,
        pl_weight=args.beta,
        batch_size=args.batch_size_GNN,
        gnn_n_layers=args.n_layers,
        n_relations=args.n_relations,
        activation=args.activation,
        gnn_epochs_per_iter=args.GNN_epochs_per_iter,
        temperature=args.temperature,
        pl_ratio=args.pl_ratio_GNN,
        intermediate_data_filepath=GNN_intermediate_data_filepath,
        ckpt_filepath=GNN_ckpt_filepath,
        pretrain_ckpt_filepath=GNN_prt_ckpt_filepath,
        train_idx=data['train_idx'],
        valid_idx=data['valid_idx'],
        test_idx=data['test_idx'],
        hard_labels=all_labels_pre_LM,
        edge_index=data['edge_index'],
        edge_type=data['edge_type'],
        num_prop=data['num_prop'],
        category_prop=data['category_prop'],
        des_tensor=data['des_tensor'],
        tweet_tensor=data['tweet_tensor'],
        SimpleHGN_att_res=args.SimpleHGN_att_res,
        att_heads=args.att_heads,
        RGT_semantic_heads=args.RGT_semantic_heads,
        gnn_hidden_dim=args.hidden_dim,
        lm_name=args.LM_model
    )
    GNNTrainer.build_model()
    GNNTrainer.train(embeddings_LM, all_labels_pre_LM)
    GNNTrainer.test(embeddings_LM, all_labels_pre_LM)
    GNNTrainer.save_results(args.experiment_name + f'/results_GNN.json')


if __name__ == '__main__':
    args = parser_args()
    main(args)

