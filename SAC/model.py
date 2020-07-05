import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn import functional as F


def weights_init_xavier(m):
    if isinstance(m, nn.Linear)\
            or isinstance(m, nn.Conv2d)\
            or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=[256, 256],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=weights_init_xavier):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units, bias=False))
        model.append(hidden_activation)
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(output_activation)

    return nn.Sequential(*model).apply(initializer)

def hard_update(target, source):
    target.load_state_dict(source.state_dict())

def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def initialize(self, inputs, batch_size):

        return dict(core_state=self.initial_state(batch_size))

    def initial_state(self, batch_size):    
        return tuple()

class TwinnedQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(TwinnedQNetwork, self).__init__()
        self.num_actions = num_actions
        self.Q1 = create_linear_network(
            num_inputs, num_actions, hidden_units=hidden_units,
            initializer=initializer)
        self.Q2 = create_linear_network(
            num_inputs, num_actions, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        #T, B, *_ = states.shape
        #states = torch.flatten(states, 0, 1)
        #states = states.float()
        #states = states.view(T * B, -1)

        q1 = self.Q1(states)
        q2 = self.Q2(states)

        #q1 = q1.view(T, B, self.num_actions)
        #q2 = q2.view(T, B, self.num_actions)

        return q1, q2

class CateoricalPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(CateoricalPolicy, self).__init__()
        self.policy = create_linear_network(
            num_inputs, num_actions, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        T, B, *_ = states.shape
        x = torch.flatten(states, 0, 1)
        x = x.float()
        x = x.view(T * B, -1)

        action_logits = self.policy(x)

        action_logits = action_logits.view(T, B, self.num_actions)

        return action_logits

    def act(self, states):
        # act with greedy policy
        action_probs = F.softmax(self.policy(states), dim=1)
        greedy_actions = torch.argmax(
            action_probs, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        # act with exploratory policy
        #action_probs = F.softmax(self.policy(states), dim=2)
        #print(states)
        action_probs = F.softmax(self.policy(states), dim=1)
        #print(action_probs)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample()#.view(-1, 1)

        # avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class SACNet(BaseNet):
    def __init__(self, observation_shape, num_actions, use_lstm=False, hidden_units=[256, 256], initializer=weights_init_xavier):
        super(SACNet, self).__init__()
        self.num_actions = num_actions
        self.policy = CateoricalPolicy(
            observation_shape[0],
            num_actions,
            hidden_units=hidden_units)
        self.critic = TwinnedQNetwork(
            observation_shape[0],
            num_actions,
            hidden_units=hidden_units)
        self.critic_target = TwinnedQNetwork(
            observation_shape[0],
            num_actions,
            hidden_units=hidden_units)

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, action_log_probs = self.policy.sample(state)
        return action, action_log_probs

    def act_greedy(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.act(state)
        return action

    def act(self, inputs, core_state=()):
        states = inputs#["frame"]
        #x = torch.flatten(x, 0, 1)
        
        #states = x.float()
        action, action_log_probs = self.explore(states)
        return (
            dict(action=action, action_log_probs=action_log_probs),
            core_state,
        )