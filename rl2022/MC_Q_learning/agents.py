from abc import ABC, abstractmethod
from collections import defaultdict
import random
from tkinter import EW
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim


class Agent(ABC):
    """Base class for Q-Learning agent

    

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        import numpy as np
        """Implement the epsilon-greedy action selection here

        

        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        
        
        q_vector = np.zeros(self.action_space.n)

        for (i, j), value in self.q_table.items():
            if i == obs:
                q_vector[j] += value

        if random.uniform(0,1) < self.epsilon:
            action = self.action_space.sample()
        else:
            action=np.argmax(q_vector)

        return action

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
    def learn(self):
        ...


class QLearningAgent(Agent):
    """
    Agent using the Q-Learning algorithm

    
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        
        import numpy as np
        
        q_vector_n_obs = np.zeros(self.action_space.n)

        for (i, j), value in self.q_table.items():
            if i == n_obs:
                q_vector_n_obs[j] += value
        best_qn = np.max(q_vector_n_obs)
        
        new_q_value = self.q_table[(obs,action)]+self.alpha*(reward+self.gamma*best_qn-self.q_table[(obs,action)])
        self.q_table[(obs,action)] = new_q_value
        
        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        
        import numpy as np
        alpha_min = 0
        alpha_max = 1
        
        #Cosine annealing
        self.alpha = alpha_min + 0.5*(alpha_max-alpha_min)*(1+np.cos(timestep/max_timestep*np.pi))
        
        
        steepness = 2
        self.alpha = (2*alpha_max)/(np.exp(steepness*timestep/max_timestep)+1)

        self.epsilon = 0.05/(timestep//1000+1)

        
        


class MonteCarloAgent(Agent):
    """
    Agent using the Monte-Carlo algorithm for training

    
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[int], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        

        :param obses (List(int)): list of received observations representing environmental states
            of trajectory (in the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        import numpy as np
        g = 0
        
        pairs=[(i,j)for i,j in zip(obses,actions)]
            
        for s in reversed(range(len(obses))):
            
            g = self.gamma*g + rewards[s]
            
            
            if (obses[s],actions[s]) not in pairs[:s]:
                
                if (obses[s],actions[s]) not in self.sa_counts.keys():
                    self.sa_counts[(obses[s],actions[s])]=0
                else:
                    self.sa_counts[(obses[s],actions[s])]+=1

                if (obses[s],actions[s]) in self.q_table.keys():
                    temp_v = self.q_table[(obses[s],actions[s])]
                else:
                    temp_v = 0
                
                new_val = (temp_v*self.sa_counts[(obses[s],actions[s])]+g)/(self.sa_counts[(obses[s],actions[s])]+1)
                updated_values[(obses[s],actions[s])] = new_val
                self.q_table[(obses[s],actions[s])] = new_val
            
                

        
        
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        import numpy as np
        init_eps = 0.65 
        steepness = 5 
        self.epsilon = (2*init_eps)/(np.exp(steepness*timestep/max_timestep)+1)
        
        
