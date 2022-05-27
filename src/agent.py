import numpy as np
import torch as T
from src.prioritized_buffer import Prioritized_Buffer
from src.duelling_deep_Q_network import Duelling_deep_Q_network


class Agent():
    def __init__(self,  n_actions, input_dims, gamma:float=0.9,  lr:float=0.01,
                 epsilon: float = 1.0, eps_min:float=0.01, eps_dec:float=5e-7,
                 mem_size:int=10000, batch_size:int=64, alpha:float=0.7, beta:float=0.4,
                 replace:int=1000, save_dir:str='models/', name:str='default'):

        '''CNN parameters'''
        self.learn_step_counter = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size

        '''Prioritized Experience Replay parameters'''
        self.alpha = alpha
        self.beta = beta

        '''Model save and update parameters'''
        self.replace_target_thresh = replace
        self.save_dir = save_dir

        '''Initialising action space, memory and Q-values'''
        self.action_space = [i for i in range(self.n_actions)]
        self.memory = Prioritized_Buffer(mem_size, self.batch_size, self.alpha, self.beta)
        self.q_eval = Duelling_deep_Q_network(self.lr, self.n_actions, input_dim=input_dims,
                               name=name, save_dir=self.save_dir)
        self.q_next = Duelling_deep_Q_network(self.lr, self.n_actions, input_dim=input_dims,
                               name=name+'.next', save_dir=self.save_dir)
        self.q_next.eval()

    '''Exploration Vs Exploitation'''
    def greedy_epsilon(self, observation):
        with T.no_grad():
            actions = T.Tensor([])
            #self.epsilon = 0 #Used only for testing
            if np.random.random() > self.epsilon:
                state = T.FloatTensor(observation).float().unsqueeze(0).to(self.q_eval.device)
                Q = self.q_eval.forward(state)
                action = np.argmax(Q.cpu().detach().numpy())
                actions = Q
            else:
                action = np.random.choice(self.action_space)
            return action, actions

    '''Store transitions and Update the target network'''
    def store_transition(self, state, state_, reward, action, done):
        self.memory.add(state, state_, reward, action, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_thresh == 0:
            print('Update target network...')
            self.q_next.load_state_dict(self.q_eval.state_dict())

    '''Function to modify beta and epsilon values'''
    def step_params(self,b):
        self.dec_epsilon()
        self.inc_beta(b)

    def dec_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def inc_beta(self, b):
        self.beta = self.beta + b if self.beta < 1 else 1

    '''Functions to save and load the network'''
    def save_models(self):
        self.q_eval.save_()
        self.q_next.save_()

    def load_models(self):
        self.q_eval.load_save()
        self.q_next.load_save()

    '''Function to compute the loss in each step'''
    def compute_loss(self):

        '''Sample the experience based on priority'''
        state, actions, reward, state_, term, weights, batch_idxs = self.memory.sample()

        states = T.FloatTensor(state).to(self.q_eval.device)
        actions = T.LongTensor(actions).type(T.int64) .to(self.q_eval.device)
        term = T.BoolTensor(term).to(self.q_eval.device)
        rewards = T.FloatTensor(reward).to(self.q_eval.device)
        states_ = T.FloatTensor(state_).to(self.q_eval.device)
        weights = T.FloatTensor(weights).to(self.q_eval.device)

        '''Determining the Target Q-value by fetching the Q-estimate from Duelling network'''
        Q_pred = self.q_eval.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_next = self.q_next.forward(states_)
        q_pred = Q_pred
        q_next = Q_next
        q_target = rewards.squeeze(1) + self.gamma * T.max(q_next, dim=1)[0]
        q_target[term] = 0.0

        '''Computing the TD-Error for PER'''
        td_errors = T.pow(q_pred - q_target, 2) * weights

        return td_errors, batch_idxs

    '''Loss mean computed for PER'''
    def learn(self):

        '''Procceeds with learning only if the Buffer has sufficient memory'''
        if not self.memory.is_sufficient():
            return
        loss, idxs = self.compute_loss()
        lossmean = loss.mean()
        self.q_eval.optimiser.zero_grad()
        lossmean.backward()

        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=0.5)
        self.q_eval.optimiser.step()
        self.learn_step_counter += 1
        for idx, td_error in zip(idxs, loss.cpu().detach().numpy()):
            self.memory.update_priorities(idx, td_error)

        return lossmean
