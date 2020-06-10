import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# Calcuate the General Advantage Estimate
def general_advantage_estimate(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * \
            values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae

        returns.insert(0, gae + values[step])

    # Advantage
    adv = np.array(returns) - values[:-1]
    return returns, (adv-np.mean(adv))/(np.std(adv)+1e-10)


def ppo_iteration(small_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // small_batch_size):
        rand_idxs = np.random.randint(0, batch_size, small_batch_size)
        yield states[rand_idxs, :], actions[rand_idxs, :], log_probs[rand_idxs, :], returns[rand_idxs, :], advantage[rand_idxs, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer, loss, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iteration(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            # Surrogate values
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0-clip_param,
                                1.0 + clip_param)*advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, action_size, input_size, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic
        mu = self.actor
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist, value
