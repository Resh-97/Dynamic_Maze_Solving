from src.display import Display
from src.generate_plots import Generate_plots
import numpy as np
import torch
import logging


def train(episodes, env, agent, beta_inc, replace_testnet, name):

    '''Creating logger object'''
    Log_Format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = "logfile_"+str(name)+".txt", filemode = "w",format = Log_Format,
                        level = logging.INFO)
    logger = logging.getLogger()

    '''Creating plotting and display objects'''
    plot = Generate_plots(name)
    display = Display()

    print('...start training...')

    for i in range(episodes):
        done = False
        observation = env.re_initialise
        display.set_visible(env.loc.copy(), env.actor_position, [])

        score = 0
        counter = 0
        while not done:
            counter +=1

            '''Obtain actions from the networks'''
            action, acts = agent.greedy_epsilon(observation)

            '''Make observations'''
            observation_, reward, done, info = env.step(action, score)

            '''Update the counters and store transitions for PER'''
            steps = len(env.actorpath)
            score += reward
            agent.store_transition(observation, observation_, reward, action, int(done))
            loss = agent.learn()

            '''Visualise the agent path'''
            observation = observation_
            display.step(env.obs2D.copy(), env.actor_position, env.actorpath, acts.data.cpu().numpy(), action,
                          score, reward, env.steps,  env.walls, env.visits, env.obs2D,name)

        agent.step_params(beta_inc)
        if i % replace_testnet == 0:
            agent.replace_target_network()

        '''plot the graphs'''
        plot.data_in(score,loss, walls=env.walls, n_stay=env.n_stay,
                    n_visits=env.visits, n_steps=steps, epsilon=agent.epsilon)

        '''Generate Logs'''
        logger.info(f'Ep {i}, Total steps {counter}, loss {loss}, score {score}, epsilon {agent.epsilon}, beta {agent.beta},\n \t\t Counters: Path len {steps} : Stayed {env.n_stay} : Walls {env.walls} : Visited {env.visits}, \n \t\t Actor path:  {env.actorpath} ')
        print(f'Ep {i}, Total steps {counter}, loss{loss} score {score}, epsilon {agent.epsilon}, beta {agent.beta}')
        print(f'    Path Len {steps} : Stayed {env.n_stay} : Walls {env.walls} : Visited {env.visits}')

        '''Save NN every 10 iterations'''
        if i > 20 and i % 20 == 0:
            agent.save_models()
            plot.live_plot()
