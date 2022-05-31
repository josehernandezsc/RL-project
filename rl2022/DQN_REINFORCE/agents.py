from abc import ABC, abstractmethod
from copy import deepcopy

import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2022.DQN_REINFORCE.networks import FCNetwork
from rl2022.DQN_REINFORCE.replay import Transition




class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """DQN agent

    

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        

        output = None
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=output
        )

        self.critics_target = deepcopy(self.critics_net)
        
        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3)
        

        
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = 1
        # ######################################### #
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        
        
        
        import numpy as np
        
        
        
        if max_timestep==300000:
            
            init_eps = 0.5
            steepness_eps = 8
        else:

            
            init_eps = 1
            steepness_eps = 8
        self.epsilon = np.clip((2*init_eps)/(np.exp(steepness_eps*timestep/max_timestep)+1),0.01,1)

        
        
        

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        
        
        import random
        if explore:
            if random.uniform(0,1) < self.epsilon:
                action = self.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(self.critics_net(torch.tensor(obs))).item()
        else:
            with torch.no_grad():
                action = torch.argmax(self.critics_net(torch.tensor(obs))).item()
            
        return action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        
        cur_state = batch[0]
        action = batch[1]
        next_state = batch[2]
        rewards = batch[3]
        done = batch[4]
        self.critics_net.train()
        self.critics_target.train(False)
        self.critics_optim.zero_grad()
        for p in self.critics_target.parameters():
            p.requires_grad = False
        
        
        q_loss = torch.sum(torch.pow(rewards+self.gamma*(1-done)*torch.reshape(torch.amax(self.critics_target(next_state),axis=1),(cur_state.shape[0],1))-
        torch.gather(self.critics_net(cur_state),1,action.long()),2))/cur_state.shape[0]
        
        
        q_loss.backward()
        for p in self.critics_target.parameters():
            p.requires_grad = False
        self.critics_optim.step()
        for p in self.critics_target.parameters():
            p.requires_grad = False
        
        
        self.update_counter += 1
        self.target_update_freq
        if self.update_counter == self.target_update_freq:
            self.critics_target = deepcopy(self.critics_net)
            self.update_counter = 0

        for p in self.critics_target.parameters():
            p.requires_grad = False
        
        
        return {"q_loss": q_loss.item()}


class Reinforce(Agent):
    """Reinforce agent

    

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n
        from torch.optim import lr_scheduler
        
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )
        
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)
        

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.policy_optim,T_max=100000,verbose=False)


        
        self.learning_rate = learning_rate
        self.gamma = gamma

        

        
        self.epsilon = 0.01
        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        
        import numpy as np
        
        init_lr = 0.001
        steepness_lr = 1
        

        self.learning_rate = (2*init_lr)/(np.exp(steepness_lr*timestep/max_timesteps)+1)

        init_eps = 0.01 
        steepness = 5 
        self.epsilon = (2*init_eps)/(np.exp(steepness*timestep/max_timesteps)+1)
        

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        

        
        import random
        if explore:
            if random.uniform(0,1) < self.epsilon:
                action = self.action_space.sample()
                
            else:
                
                probs = self.policy(torch.tensor(obs))
                m = Categorical(probs)
                action=m.sample().item()
                
        else:
            
            action = self.policy(torch.tensor(obs)).multinomial(num_samples=1,replacement=True).item()
        return action

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        
        self.policy.train()
        
        
        loss = []
        
        g = 0
        for s in reversed(range(len(rewards))):
            probs = self.policy(torch.tensor(observations[s]))
            m = Categorical(probs)
            g = rewards[s]+self.gamma*g
            loss.append(-g*m.log_prob(torch.tensor(actions[s])))
            
        loss = torch.stack(loss).sum()/len(rewards)
        
        
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        self.scheduler.step()
        p_loss=loss.item()
        
        
        return {"p_loss": p_loss}
