from torch.distributions import Categorical
from torch import nn


class Agent(nn.Module):
    '''Actor Critic Network ...'''
    def __init__(self, in_features_act, in_features_cri, out_features_act, out_features_cri, num_layers, layer_size):
        super(Agent, self).__init__()

        self.in_features_act = in_features_act
        self.in_features_cri = in_features_cri

        self.out_features_act = out_features_act
        self.out_features_cri = out_features_cri

        self.num_layers = num_layers
        self.layer_size = layer_size

        self.actor = self.initialize_mlp(in_features_act, out_features_act)
        self.critic = self.initialize_mlp(in_features_cri, out_features_cri)


    def get_value(self, obs):
        return self.critic(obs[:, 0 ,3:6]).squeeze(-1)

    
    def get_logits(self, obs, action_mask=None):
        logits = self.actor(obs).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        return logits

    
    def get_action_and_value(self, obs, action=None, action_mask=None):
        dist = Categorical(logits=self.get_logits(obs, action_mask=action_mask))
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.get_value(obs)


    def initialize_mlp(self, in_features, out_features):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(in_features, self.layer_size))
                layers.append(nn.ReLU())
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(self.layer_size, out_features))
            else:
                layers.append(nn.Linear(self.layer_size, self.layer_size))
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)