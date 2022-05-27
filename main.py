from src.environment import Environment
from src.agent import Agent
from src.display import Display
from src.train import train
from src.evaluate import test
import os

'''setting environment parameter'''
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def run(test_flag=False, train_flag=True, lr=0.01, epsilon=0.9,
        gamma=0.9, episodes=100,filename = '5x5model_test', name='default', ep_min=0.01, ep_dec = 1e-4, batch_size=128, beta_inc=0.1):

    '''Initialising parameters for CNN and exploration'''
    input_dims = [303]
    output_dims = 5
    replace_network = 3
    memsize = 100000
    batch_size = batch_size
    epsilon_min = ep_min
    epsilon_dec = ep_dec

    '''Creating environment and agent objects'''
    env = Environment()
    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr,
                  input_dims=input_dims, n_actions=output_dims, mem_size=memsize, eps_min=epsilon_min,
                  batch_size=batch_size, eps_dec=epsilon_dec, replace=replace_network, name=name)


    '''Begin training'''
    if train_flag:
        train(episodes, env, agent, beta_inc, replace_network, filename)

    '''Evaluation'''
    if test_flag:
        agent.load_models()
        test(env, agent, beta_inc, filename)



if __name__ == '__main__':
    run(test_flag=True, train_flag = False, episodes=600,
            lr=0.001, epsilon=0.8, gamma=0.999, filename = '37x37model_test',
           name='default', batch_size=64, ep_min=0.00033, beta_inc=0.01, ep_dec=0.005)
