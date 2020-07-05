import threading
import typing

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)





def calc_policy_loss(model, states, weights, alpha):

    # (log of) probabilities to calculate expectations of Q and entropies
    _, action_probs, log_action_probs = model.policy.sample(states)
    with torch.no_grad():
        # Q for every actions to calculate expectations of Q
        q1, q2 = model.critic(states)
    
    # expectations of entropies
    entropies = -torch.sum(
        action_probs * log_action_probs, dim=1, keepdim=True)
    # expectations of Q
    q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

    # Policy objective is maximization of (Q + alpha * entropy) with
    # priority weights.
    policy_loss = (weights * (- q - alpha * entropies)).mean()

    return policy_loss, entropies

def calc_current_q(model, states, actions):
    curr_q1, curr_q2 = model.critic(states)
    curr_q1 = curr_q1.gather(1, actions.unsqueeze(1).long())
    curr_q2 = curr_q2.gather(1, actions.unsqueeze(1).long())

    return curr_q1, curr_q2

def calc_target_q(model, rewards, next_states, dones, alpha, gamma_n):
    with torch.no_grad():
        _, action_probs, log_action_probs =\
            model.policy.sample(next_states)
        #next_q1, next_q2 = model.critic_target(next_states)
        next_q1, next_q2 = model.critic_target(next_states)
        next_q = (action_probs * (
            torch.min(next_q1, next_q2) - alpha * log_action_probs
            )).mean(dim=1, keepdim=True)
    
    target_q = rewards.view_as(next_q) + (1.0 - dones.view_as(next_q)) * gamma_n * next_q
    
    return target_q
    
def calc_critic_loss(model, states, actions, rewards, dones, next_states, weights, alpha, gamma_n):
    curr_q1, curr_q2 = calc_current_q(model, states, actions)
    target_q = calc_target_q(model, rewards, next_states, dones, alpha, gamma_n)

    # TD errors for updating priority weights
    errors = torch.abs(curr_q1.detach() - target_q)
    # We log means of Q to monitor training.
    mean_q1 = curr_q1.detach().mean().item()
    mean_q2 = curr_q2.detach().mean().item()
    
    # Critic loss is mean squared TD errors with priority weights.
    q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
    q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
    return q1_loss, q2_loss, errors, mean_q1, mean_q2

def calc_entropy_loss(log_alpha, target_entropy, entropy, weights):
    # Intuitively, we increse alpha when entropy is less than target
    # entropy, vice versa.
    entropy_loss = -torch.mean(
        log_alpha * (target_entropy - entropy).detach()
        * weights)
    return entropy_loss

def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()


def unbatch(flags, batch):
    states = batch["frame"][:flags.unroll_length]

    actions = batch["action"][1:]
    action_log_probs = batch["action_log_probs"][1:]

    rewards = batch["reward"][1:]
    if flags.reward_clipping == "abs_one":
        rewards = torch.clamp(rewards, -1, 1)
    elif flags.reward_clipping == "none":
        rewards = rewards
    dones = batch["done"][1:].float()
    
    next_states = batch["frame"][1:]
    

    return states, actions, rewards, next_states, dones, action_log_probs

def learn(
    flags,
    #actor_model,
    learner_model,
    batch,
    #initial_agent_state,
    optim,
    #step,
    lock=threading.Lock(),  # noqa: B008
    weights=1,
):
    """Performs a learning (optimization) step."""
    with lock:
        states, actions, rewards, next_states, dones, action_log_probs = unbatch(flags, batch)
        
        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            calc_critic_loss(learner_model, states, actions, rewards, dones, next_states, weights, optim.alpha, flags.discounting**flags.multi_step)
        policy_loss, entropies = calc_policy_loss(learner_model, states, weights, optim.alpha)
        
        update_params(
            optim.q1_optim, learner_model.critic.Q1, q1_loss, flags.grad_norm_clipping)
        update_params(
            optim.q2_optim, learner_model.critic.Q2, q2_loss, flags.grad_norm_clipping)
        update_params(
            optim.policy_optim, learner_model.policy, policy_loss, flags.grad_norm_clipping)

        if optim.entropy_tuning:
            entropy_loss = calc_entropy_loss(optim.log_alpha, optim.target_entropy, entropies, weights)
            update_params(optim.alpha_optim, None, entropy_loss)
            optim.alpha = optim.log_alpha.exp()
        
        #actor_model.policy.load_state_dict(learner_model.policy.state_dict())

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "0_episode_returns": tuple(episode_returns.cpu().numpy()),
            "1_mean_episode_return": torch.mean(episode_returns).item(),
            "2_q1_loss": q1_loss.item(),
            "3_q2_loss": q2_loss.item(),
            "4_policy_loss": policy_loss.item(),
            "5_entropy_loss": entropy_loss.item(),
            "6_alpha": optim.alpha.item(),
            "7_entoropy": entropies.mean().item()
        }
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        action_log_probs=dict(size=(T + 1, num_actions), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


class optimizer:
    def __init__(self, flags, model, entropy_tuning=True):
        self.policy_optim = Adam(model.policy.parameters(), lr=flags.learning_rate)
        self.q1_optim = Adam(model.critic.Q1.parameters(), lr=flags.learning_rate)
        self.q2_optim = Adam(model.critic.Q2.parameters(), lr=flags.learning_rate)

        self.entropy_tuning = entropy_tuning
        self.device = flags.device

        if self.entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                flags.action_shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=flags.learning_rate)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)


