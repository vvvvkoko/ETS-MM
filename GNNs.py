import torch.nn as nn
import torch
from torch_geometric.nn import RGCNConv, HGTConv, SAGEConv, GATConv, GCNConv

class SAGE2(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = 160
        self.num_prop_size = 6
        self.num_category_size = 11

        self.linear_relu_des = nn.Sequential(
            nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5)),
            nn.LeakyReLU()
        )

        self.linear_relu_text = nn.Sequential(
            nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5)),
            nn.LeakyReLU()
        )

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(self.num_prop_size, int(self.hidden_dim/5)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_categpry = nn.Sequential(
            nn.Linear(self.num_category_size, int(self.hidden_dim/5)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU()
        )

        self.dropout = nn.Dropout(model_config['dropout'])
        self.conv1 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU()
        )

        self.linear_output2 = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, num_prop, num_category, des_tensor, tweet_tensor):
        d = self.dropout(self.linear_relu_des(des_tensor))
        tw = self.dropout(self.linear_relu_tweet(tweet_tensor))
        t = self.dropout(self.linear_relu_text(x))
        n = self.dropout(self.linear_relu_num_prop(num_prop))
        c = self.dropout(self.linear_relu_num_categpry(num_category))
        x = torch.cat((n,c,d,tw,t), dim=1)
        x = self.dropout(self.linear_relu_input(x))
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        em = self.linear_relu_output1(x)
        x = self.linear_output2(em)
        return x, em


class SAGE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = 32*5
        self.num_prop_size = 6
        self.num_category_size = 11

        self.linear_relu_tweet = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5))
        self.linear_relu_num_prop = nn.Linear(self.num_prop_size, int(self.hidden_dim/5))
        self.linear_relu_num_categpry = nn.Linear(self.num_category_size, int(self.hidden_dim/5))
        self.linear_relu_des = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_text = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5))

        self.linear_relu_input = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.conv1 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.linear_relu_output1 = nn.Linear(self.hidden_dim, 80)
        self.linear_output2 = nn.Linear(80, 2)

        self.ReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(model_config['dropout'])

    def forward(self, pre_x, x, edge_index, edge_type, num_prop, num_category, des_tensor, tweet_tensor):
        n = self.dropout(self.ReLU(self.linear_relu_num_prop(num_prop)))
        c = self.dropout(self.ReLU(self.linear_relu_num_categpry(num_category)))
        d = self.dropout(self.ReLU(self.linear_relu_des(des_tensor)))
        tw = self.dropout(self.ReLU(self.linear_relu_text(tweet_tensor)))
        pre_t = self.dropout(self.ReLU(self.linear_relu_tweet(pre_x)))
        x = torch.cat((n,c,d,tw,pre_t), dim=1)
        x = self.dropout(self.ReLU(self.linear_relu_input(x)))
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        em = self.dropout(self.ReLU(self.linear_relu_output1(x)))
        x = self.linear_output2(em)
        return x, em


class RGCN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = 32*5
        self.num_prop_size = 6
        self.num_category_size = 11

        self.linear_relu_num_prop = nn.Linear(self.num_prop_size, int(self.hidden_dim/5))
        self.linear_relu_num_categpry = nn.Linear(self.num_category_size, int(self.hidden_dim/5))
        self.linear_relu_des = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5))
        self.linear_relu_text = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5))
        self.linear_relu_tweet = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim/5))
        self.linear_relu_input = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.conv1 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=2)
        self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=2)
        self.linear_relu_output1 = nn.Linear(self.hidden_dim, 80)
        self.linear_output2 = nn.Linear(80, 2)

        self.ReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(model_config['dropout'])

    def forward(self, pre_x, x, edge_index, edge_type, num_prop, num_category, des_tensor, tweet_tensor):
        n = self.dropout(self.ReLU(self.linear_relu_num_prop(num_prop)))
        c = self.dropout(self.ReLU(self.linear_relu_num_categpry(num_category)))
        d = self.dropout(self.ReLU(self.linear_relu_des(des_tensor)))
        tw = self.dropout(self.ReLU(self.linear_relu_text(tweet_tensor)))
        pre_t = self.dropout(self.ReLU(self.linear_relu_tweet(pre_x)))
        x = torch.cat((n,c,d,tw,pre_t), dim=1)
        x = self.linear_relu_input(x)
        x = self.dropout(self.ReLU(x))
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        em = self.dropout(self.ReLU(self.linear_relu_output1(x)))
        x = self.linear_output2(em)
        return x, em


class HGT(nn.Module):
    def __init__(self, model_config):
        super(HGT, self).__init__()
        self.hidden_dim = 32*5
        self.des_dim = 768
        self.tweet_dim = 768
        self.lm_dim = 768
        self.num_prop_size = 6
        self.num_category_size = 11
        self.dropout = 0.5

        self.linear_relu_num_prop = nn.Linear(self.num_prop_size, int(self.hidden_dim / 5))
        self.linear_relu_num_categpry = nn.Linear(self.num_category_size, int(self.hidden_dim / 5))
        self.linear_relu_des = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_text = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_tweet = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_input = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.HGT_layer1 = HGTConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                  metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))
        self.HGT_layer2 = HGTConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                  metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))

        self.linear_relu_output1 = torch.nn.Linear(self.hidden_dim, 80)
        self.linear_output2 = nn.Linear(80, 2)

        self.dropout = nn.Dropout(model_config['dropout'])
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, pre_x, x, edge_index, edge_type, num_prop, num_category, des_tensor, tweet_tensor):
        n = self.dropout(self.ReLU(self.linear_relu_num_prop(num_prop)))
        c = self.dropout(self.ReLU(self.linear_relu_num_categpry(num_category)))
        d = self.dropout(self.ReLU(self.linear_relu_des(des_tensor)))
        tw = self.dropout(self.ReLU(self.linear_relu_text(tweet_tensor)))
        pre_t = self.dropout(self.ReLU(self.linear_relu_tweet(pre_x)))
        x = torch.cat((n, c, d, tw, pre_t), dim=1)
        x = self.dropout(self.ReLU(self.linear_relu_input(x)))
        x_dict = {'user': x}
        edge_index_dict = {('user','follower','user'): edge_index[:, edge_type==0],
                           ('user','following','user'): edge_index[:, edge_type==1]}
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        x_dict = self.HGT_layer2(x_dict, edge_index_dict)
        em = self.dropout(self.ReLU(self.linear_relu_output1(x_dict['user'])))
        x = self.linear_output2(em)
        return x, em


class GAT(nn.Module):
    def __init__(self, model_config):
        super(GAT, self).__init__()
        self.hidden_dim = 32 * 5
        self.des_dim = 768
        self.tweet_dim = 768
        self.lm_dim = 768
        self.num_prop_size = 6
        self.num_category_size = 11
        self.dropout = 0.5

        self.linear_relu_num_prop = nn.Linear(self.num_prop_size, int(self.hidden_dim / 5))
        self.linear_relu_num_categpry = nn.Linear(self.num_category_size, int(self.hidden_dim / 5))
        self.linear_relu_des = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_text = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_tweet = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 5))
        self.linear_relu_input = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.conv1 = GATConv(self.hidden_dim, self.hidden_dim)
        self.conv2 = GATConv(self.hidden_dim, self.hidden_dim)

        self.linear_relu_output1 = torch.nn.Linear(self.hidden_dim, 80)
        self.linear_output2 = nn.Linear(80, 2)

        self.dropout = nn.Dropout(model_config['dropout'])
        self.ReLU = nn.LeakyReLU()

    def forward(self, pre_x, x, edge_index, edge_type, num_prop, num_category, des_tensor, tweet_tensor):
        n = self.dropout(self.ReLU(self.linear_relu_num_prop(num_prop)))
        c = self.dropout(self.ReLU(self.linear_relu_num_categpry(num_category)))
        d = self.dropout(self.ReLU(self.linear_relu_des(des_tensor)))
        tw = self.dropout(self.ReLU(self.linear_relu_text(tweet_tensor)))
        pre_t = self.dropout(self.ReLU(self.linear_relu_tweet(pre_x)))
        x = torch.cat((n, c, d, tw,pre_t), dim=1)
        x = self.dropout(self.ReLU(self.linear_relu_input(x)))
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        em = self.dropout(self.ReLU(self.linear_relu_output1(x)))
        x = self.linear_output2(em)
        return x, em


class GCN(nn.Module):
    def __init__(self, model_config):
        super(GCN, self).__init__()
        self.hidden_dim = 32 * 6
        self.des_dim = 768
        self.tweet_dim = 768
        self.lm_dim = 768
        self.num_prop_size = 5
        self.num_category_size = 1
        self.dropout = 0.5

        self.linear_relu_tweet = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 6))
        self.linear_relu_tweet_tr = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 6))
        self.linear_relu_num_prop = nn.Linear(self.num_prop_size, int(self.hidden_dim / 6))
        self.linear_relu_num_categpry = nn.Linear(self.num_category_size, int(self.hidden_dim / 6))
        self.linear_relu_des = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 6))
        self.linear_relu_text = nn.Linear(model_config['lm_input_dim'], int(self.hidden_dim / 6))
        self.linear_relu_input = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.conv1 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.linear_relu_output1 = torch.nn.Linear(self.hidden_dim, 96)
        self.linear_output2 = nn.Linear(96, 2)

        self.dropout = nn.Dropout(model_config['dropout'])
        self.ReLU = nn.LeakyReLU()

    def forward(self, pre_x, x, edge_index, edge_type, num_prop, num_category, des_tensor, tweet_tensor):
        d = self.dropout(self.ReLU(self.linear_relu_des(des_tensor)))
        pre_t = self.dropout(self.ReLU(self.linear_relu_tweet(pre_x)))
        t = self.dropout(self.ReLU(self.linear_relu_tweet_tr(x)))
        n = self.dropout(self.ReLU(self.linear_relu_num_prop(num_prop)))
        c = self.dropout(self.ReLU(self.linear_relu_num_categpry(num_category)))
        tw = self.dropout(self.ReLU(self.linear_relu_text(tweet_tensor)))
        x = torch.cat((n, c, d, tw, pre_t, t), dim=1)
        x = self.dropout(self.ReLU(self.linear_relu_input(x)))
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        em = self.dropout(self.ReLU(self.linear_relu_output1(x)))
        x = self.linear_output2(em)
        return x, em



