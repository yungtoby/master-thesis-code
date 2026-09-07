from torch.distributions import Categorical
from torch import nn
import torch as t


class Agent(nn.Module):
    '''Actor Critic Network ...'''
    def __init__(self, observation_dim, in_features_act, out_features_act, out_features_cri, num_layers, layer_size, critic_feature_mode="global3"):
        super(Agent, self).__init__()

        self.observation_dim = observation_dim
        self.critic_feature_mode = critic_feature_mode

        self.in_features_act = in_features_act

        self.out_features_act = out_features_act
        self.out_features_cri = out_features_cri

        self.num_layers = num_layers
        self.layer_size = layer_size

        self.actor = self.initialize_mlp(in_features_act, out_features_act)
        self.critic = self.initialize_mlp(self.critic_input_dim(observation_dim, critic_feature_mode), out_features_cri)


    @staticmethod
    def critic_input_dim(observation_dim, mode):
        if mode == "global3":
            return 3

        if mode == "mean_max_global":
            local_dim = observation_dim - 3
            return 2 * local_dim + 3

        raise ValueError(f"Unknown critic_feature_mode={mode}")


    def get_critic_features(self, obs):
        if self.critic_feature_mode == "global3":
            return obs[:, 0, -3:]

        local = obs[..., :-3]
        global_features = obs[:, 0, -3:]

        return t.cat(
            [
                local.mean(dim=1),
                local.max(dim=1).values,
                global_features,
            ],
            dim=-1,
        )


    def get_value(self, obs):
        return self.critic(self.get_critic_features(obs)).squeeze(-1)

    
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