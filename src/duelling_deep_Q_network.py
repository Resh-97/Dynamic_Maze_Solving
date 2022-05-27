import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Duelling_deep_Q_network(nn.Module):

    def __init__(self, lr, n_actions, name, input_dim, save_dir):

        super(Duelling_deep_Q_network, self).__init__()
        '''Initialising'''
        self.save_dir = save_dir
        self.save_file = os.path.join(self.save_dir, name)

        '''Feature extrcation'''
        self.features = nn.Sequential(
            nn.Linear(*input_dim, 1024),
            nn.ReLU()
        )

        '''Value Stream'''
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)
        )

        '''Action advantage stream'''
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_actions)
        )

        '''Setting optimiser and Loss function'''
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    '''Forward pass to compute next Q-value'''
    def forward(self, state):
        features = self.features(state)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        Q = T.add(V, (A - A.mean()))
        return Q

    '''Functions to save and load network'''
    def save_(self):
        T.save(self.state_dict(), self.save_file)

    def load_save(self):
        self.load_state_dict(T.load(self.save_file))
