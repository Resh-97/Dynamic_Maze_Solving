from src.sumtree import SumTree
import numpy as np
import random

class Prioritized_Buffer:

    def __init__(self, max_size, batch_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.current_length = 0
        self.batch_size = batch_size

    ''' Function to store transitions in the replay buffer'''
    def add(self, state, next_state, reward,  action, done):
        priority = 1.0 if self.current_length == 0 else self.sum_tree.tree.max()
        self.current_length = self.current_length + 1
        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)

    '''Function to sample the Experience based on priority'''
    def sample(self):
        batch_idx, batch, IS_weights = [], [], []

        '''Segment the sumtree (saved expeirences) into chunks equal to batchsize.'''
        segment = self.sum_tree.total() / self.batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)
            batch_idx.append(idx)
            batch.append(data)

            '''Compute the Pri0rity probability'''
            prob = p / p_sum

            '''Compute the importance-sampling weight'''
            IS_weight = (self.sum_tree.total() * prob) ** (-self.beta)
            IS_weights.append(IS_weight)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        '''Modify the PER output based on priority'''
        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, np.array(reward_batch), next_state_batch, done_batch, IS_weights, batch_idx

    '''Function to update priority based on TD-error'''
    def update_priorities(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    '''Checking if sufficient memory available'''
    def is_sufficient(self):
        return self.current_length > self.batch_size
