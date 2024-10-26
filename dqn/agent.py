# Inspired from https://github.com/raillab/dqn
from gym import spaces
import numpy as np

from dqn.model_nature import DQN as DQN_nature
from dqn.model_neurips import DQN as DQN_neurips
from dqn.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 use_double_dqn,
                 lr,
                 batch_size,
                 gamma,
                 device=torch.device("cpu" ),
                 dqn_type="neurips"):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma

        if(dqn_type=="neurips"):
            DQN = DQN_neurips
        else:
            DQN = DQN_nature

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters()
            , lr=lr)        
        ## self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.device = device

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.policy_network(next_states).max(1)
                max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_states)
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # passing states (batch_size,C,H,W) through policy_network returns (B,A) i.e. state value for each action
        # but we are interested in state values corresponding to chosen actions
        # so we want to pick particular value from each row of (B,A) where A = number of actions
        # gather method achieves that. You tell gather method along which dimension you want to pick values
        # in our case we want to pick values along 1st dimension of input_q_values tensor
        # second argument to gather method is the indices to pick up, its shape must match the shape of input_q_values
        # for instance input_q_values[0,:] will have 6 values to choose from. method gather will pick up the value based
        # on actions.unsqueeze(1)[0] which will yield a particular action index from 0 to 5
        # actions.unsqueeze(1) will convert tensor from (batch_size,) to (batch_size,1)
        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()
