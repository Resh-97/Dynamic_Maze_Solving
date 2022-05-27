import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class Generate_plots:

    def __init__(self, name):

        '''Initialising'''
        self.name = name
        self.scores = []
        self.loss = []
        self.average_score = []
        self.eps_history = []

        self.walls_list = []
        self.n_stay_list = []
        self.visits_list = []
        self.steps_list = []

    ''' Function to Obtain values for each episode from Train.py'''
    def data_in(self, score, loss, walls=-1, n_stay=-1, n_visits=-1, n_steps=-1, epsilon=-1):
        self.scores.append(score)
        self.loss.append(loss)
        self.average_score.append(np.mean(self.scores[-100:]))
        if walls > -1:
            self.walls_list.append(walls)
        if n_stay > -1:
            self.n_stay_list.append(n_stay)
        if n_visits > -1:
            self.visits_list.append(n_visits)
        if n_steps > -1:
            self.steps_list.append(n_steps)
        self.eps_history.append(epsilon)
        with open('json_data/scores'+str(self.name)+'.json', 'w') as jf:
            json.dump(self.scores, jf)

    @property
    def show(self):
        plt.show()

    def plot_socre(self):
        plt.figure()
        plt.plot(self.scores, color='red', label='score', alpha=0.5)
        plt.xlabel('Epochs')
        plt.legend()
        plt.ylabel('Loss')
        plt.savefig('plots/Loss'+str(self.name)+'.png')
        plt.clf()
        plt.close('all')

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss, color='red', label='Score', alpha=0.5)
        plt.plot(self.average_score, color='blue', label='Average', alpha=0.5)
        plt.xlabel('Epochs')
        plt.legend()
        plt.ylabel('Avg Score')
        plt.savefig('plots/Score'+str(self.name)+'.png')
        plt.clf()
        plt.close('all')

    def plot_epsilon(self):
        plt.figure()
        plt.plot(self.eps_history, color='red', label='epsilon')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Episilon')
        plt.savefig('plots/Epsilon'+str(self.name)+'.png')
        plt.clf()
        plt.close('all')

    def plt_pathlength(self):
        plt.figure()
        plt.plot(self.steps_list, color='red', label='epsilon')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Steps')
        plt.savefig('plots/Path'+str(self.name)+'.png')
        plt.clf()
        plt.close('all')

    def live_plot(self):
        mpl.use("agg")
        print('Plotting ...')
        self.plot_socre()
        self.plot_loss()
        self.plt_pathlength()

        plt.clf()
        plt.close('all')
