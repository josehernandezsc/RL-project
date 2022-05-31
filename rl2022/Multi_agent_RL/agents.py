from abc import ABC, abstractmethod
from collections import defaultdict
from ntpath import join
import random
from typing import List, Dict, DefaultDict

from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
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
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        

        :return (List[int]): index of selected action for each agent
        """
        import numpy as np
        
        q_vector_1 = np.zeros(self.action_spaces[0].n)

        for j, value in self.q_tables[0].items():
            
            q_vector_1[j] += value

        q_vector_2 = np.zeros(self.action_spaces[1].n)

        for j, value in self.q_tables[1].items():
            
            q_vector_2[j] += value
        
        
        
        if random.uniform(0,1) < self.epsilon:
            
            action = self.action_spaces.sample()
            
            
        else:
            action = (np.argmax(q_vector_1),np.argmax(q_vector_2))
            
        
            
        
        actions = [i for i in action]
        
        
        return actions

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []
        
        import numpy as np
        
        for a in range(self.num_agents):

            q_vector_n_obs = np.zeros(self.action_spaces[a].n)
            
            for j, value in self.q_tables[a].items():
                q_vector_n_obs[j] += value
            

            best_qn = np.max(q_vector_n_obs)
                
            new_q_value = self.q_tables[a][actions[a]]+self.learning_rate*(rewards[a]+self.gamma*best_qn-self.q_tables[a][actions[a]])
            self.q_tables[a][actions[a]] = new_q_value
            updated_values.append(new_q_value)
            
            
            
        
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        
        import numpy as np
        
        
        init_lr = 0.001
        steepness = 3
        self.learning_rate = (2*init_lr)/(np.exp(steepness*timestep/max_timestep)+1)
        


        init_eps = 0.9
        steepness = 2
        self.epsilon = np.clip((2*init_eps)/(np.exp(steepness*timestep/max_timestep)+1),0.1,1)
        


class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]
        
        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 

    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        

        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        import numpy as np
        
        
        
        if bool(self.models[0]) == False and bool(self.models[1]) == False:
            
            for ac1 in range(self.n_acts[0]):
                for ac2 in range(self.n_acts[1]):
                    self.models[0][(ac2,)] = 0
                    self.models[1][(ac1,)] = 0
        
        ev = np.zeros((self.num_agents,self.n_acts[0]))
        #Assuming there are only 2 agents
        
        

        max_ev=0
        for ac1 in range(self.n_acts[0]):
            max_ev=0
            for ac2 in range(self.n_acts[1]):
                max_ev += self.models[0][(ac2,)]
            for ac2 in range(self.n_acts[1]):
                
                ev[0,ac1]+=self.models[0][(ac2,)]*self.q_tables[0][(ac1,ac2)]/max(1, max_ev)

        

        
        max_ev = 0
        for ac2 in range(self.n_acts[1]):
            max_ev=0
            for ac1 in range(self.n_acts[0]):
                max_ev += self.models[1][(ac1,)]
            for ac1 in range(self.n_acts[0]):
                
                ev[1,ac2]+=self.models[1][(ac1,)]*self.q_tables[1][(ac1,ac2)]/max(1, max_ev)
        
        
        
        if random.uniform(0,1) < self.epsilon:
            action = self.action_spaces.sample()
            
        else:
            action = np.argmax(ev,axis=1)
            
            
            

        
        joint_action = [i for i in action]
        
        return joint_action

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        import numpy as np


        

        self.models[1][(actions[0],)]+=1 #a1 according to a2
        self.models[0][(actions[1],)]+=1 #a2 according to a1
        
        ev = np.zeros((self.num_agents,self.n_acts[0]))
        #Assuming there are only 2 agents
        
        
        
        max_ev=0
        for ac1 in range(self.n_acts[0]):
            max_ev=0
            for ac2 in range(self.n_acts[1]):
                max_ev += self.models[0][(ac2,)]
            for ac2 in range(self.n_acts[1]):
                ev[0,ac1]+=self.models[0][(ac2,)]*self.q_tables[0][(ac1,ac2)]/max(1, max_ev)

        print(dones)

        
        max_ev = 0
        for ac2 in range(self.n_acts[1]):
            max_ev=0
            for ac1 in range(self.n_acts[0]):
                max_ev += self.models[1][(ac1,)]
            for ac1 in range(self.n_acts[0]):
                
                ev[1,ac2]+=self.models[1][(ac1,)]*self.q_tables[1][(ac1,ac2)]/max(1, max_ev)
                
                
        
        

        new_val = rewards[0] + self.gamma*np.max(ev[0,:]) - self.q_tables[0][(actions[0],actions[1])]
        self.q_tables[0][(actions[0],actions[1])] += self.learning_rate*new_val
        updated_values.append(self.q_tables[0][(actions[0],actions[1])])
        new_val = rewards[1] + self.gamma*np.max(ev[1,:]) - self.q_tables[1][(actions[0],actions[1])]
        self.q_tables[1][(actions[0],actions[1])] += self.learning_rate*new_val
        updated_values.append(self.q_tables[1][(actions[0],actions[1])])
        
        
        
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        
        import numpy as np
        
        
        init_lr = 0.001 #0.001
        steepness = 3 #2
        self.learning_rate = (2*init_lr)/(np.exp(steepness*timestep/max_timestep)+1)
        
        init_eps = 0.6
        steepness = 4
        self.epsilon = np.clip((2*init_eps)/(np.exp(steepness*timestep/max_timestep)+1),0.05,1)
        
