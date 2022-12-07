import numpy as np
import random
import copy
from collections import namedtuple, deque



import torch
import torch.nn.functional as F
import torch.optim as optim

import importlib
import model
importlib.reload(model)
#from model import Actor, Critic


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256      # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, joint_state_size, joint_action_size, random_seed, noise_scale=0.2, noise_decay=0.9999, lr_actor = 1e-4, lr_critic = 1e-4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.joint_state_size=joint_state_size
        self.joint_action_size=joint_action_size

        # Actor Network (w/ Target Network)
        self.actor_local = model.Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = model.Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = model.Critic(joint_state_size, joint_action_size, random_seed).to(device)
        self.critic_target = model.Critic(joint_state_size, joint_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        
        self.noise_scale=noise_scale
        self.noise_decay=noise_decay
        

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += np.random.randn()*self.noise_scale
        return np.clip(action, -1, 1)
    
    def act_target(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        self.actor_target.train()
        if add_noise:
            action += np.random.randn()*self.noise_scale
        return np.clip(action, -1, 1)

    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states=np.array([e.state for e in experiences if e is not None])
        states= [torch.from_numpy(states[:,i,:]).float().to(device) for i in range(self.num_agents)]
        
        actions=np.array([e.action for e in experiences if e is not None])
        actions= [torch.from_numpy(actions[:,i,:]).float().to(device) for i in range(self.num_agents)]
        
        rewards=np.array([e.reward for e in experiences if e is not None])
        rewards= [torch.from_numpy(rewards[:,i].reshape(-1,1)).float().to(device) for i in range(self.num_agents)]
        
        next_states=np.array([e.next_state for e in experiences if e is not None])
        next_states= [torch.from_numpy(next_states[:,i,:]).float().to(device) for i in range(self.num_agents)]
        
        dones=np.array([e.done for e in experiences if e is not None])
        dones= [torch.from_numpy(dones[:,i].reshape(-1,1).astype(float)).float().to(device) for i in range(self.num_agents)]

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class MultiAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, update_every=20, learn_n_times=10, noise_scale=0.2, noise_decay=0.9999, lr_actor = 1e-4, lr_critic = 1e-4, actor_checkpoint_path='actor_checkpoint', critic_checkpoint_path='critic_checkpoint'    ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)


        #Replay memory
        self.memory = ReplayBuffer(num_agents, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        #Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.update_every=update_every
        self.learn_n_times=learn_n_times
        
        self.num_agents=num_agents
        
        self.actor_checkpoint_path=actor_checkpoint_path
        self.critic_checkpoint_path=critic_checkpoint_path
        
        
        #Individual agents
        self.agents=[Agent(state_size=state_size, action_size=action_size, joint_state_size=state_size*num_agents, joint_action_size=action_size*num_agents, random_seed=random_seed, noise_scale=noise_scale, noise_decay=noise_decay, lr_actor = lr_actor, lr_critic = lr_critic ) for _ in range(num_agents)]
        
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
#         if (np.array(reward)!=0).any():
#             self.memory.add(state, action, reward, next_state, done)
#         elif np.random.random() < 0.1:
#             self.memory.add(state, action, reward, next_state, done)
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for i in range(self.learn_n_times):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)



    

    def learn(self, experiences,  gamma):
        collective_states, collective_actions, collective_rewards, collective_next_states, collective_dones =experiences
        joint_states=torch.cat(collective_states, dim=1)
        joint_actions=torch.cat(collective_actions, dim=1)
        joint_next_states=torch.cat(collective_next_states, dim=1)
        


        for agent, states, actions, rewards, next_states,  dones in zip(self.agents, collective_states, collective_actions, collective_rewards, collective_next_states, collective_dones):
             # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            #actions_next = agent.actor_target(next_states)
            joint_actions_next=torch.cat([agent1.actor_target(next_states1) for agent1, next_states1 in zip(self.agents, collective_next_states)],dim=1)
            Q_targets_next = agent.critic_target(joint_next_states, joint_actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = agent.critic_local(joint_states, joint_actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            #actions_pred = agent.actor_local(states)
            joint_actions_pred=torch.cat([agent1.actor_local(states1) for agent1, states1 in zip(self.agents, collective_states)],dim=1)
            actor_loss = -agent.critic_local(joint_states, joint_actions_pred).mean()
            # Minimize the loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic_local, agent.critic_target, TAU)
            agent.soft_update(agent.actor_local, agent.actor_target, TAU)  
        
            agent.noise_scale *= agent.noise_decay
            
        
        

        
    def act(self, states, add_noise=True):
        
        
        actions=np.vstack([agent.act(state.reshape(1,-1), add_noise=add_noise) for state, agent in zip(states, self.agents)])
        return actions
    
    def act_target(self, states, add_noise=False):
        actions=np.vstack([agent.act_target(state.reshape(1,-1), add_noise=add_noise) for state, agent in zip(states, self.agents)])
        return actions
    
    def checkpoint(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), '{}_{}.pth'.format(self.actor_checkpoint_path, i))
            torch.save(agent.critic_local.state_dict(), '{}_{}.pth'.format(self.critic_checkpoint_path, i))
    def load(self, actor_checkpoint_path=None, critic_checkpoint_path=None):
        if actor_checkpoint_path==None:
            actor_checkpoint_path=self.actor_checkpoint_path
        if critic_checkpoint_path==None:
            critic_checkpoint_path=self.critic_checkpoint_path
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('{}_{}.pth'.format(actor_checkpoint_path, i), map_location=lambda storage, loc: storage))
            agent.critic_local.load_state_dict(torch.load('{}_{}.pth'.format(critic_checkpoint_path, i), map_location=lambda storage, loc: storage)) 
            

        
