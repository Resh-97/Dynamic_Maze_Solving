from src.display import Display
from src.generate_plots import Generate_plots
import numpy as np
import torch
import logging

def test(env, agent, beta_inc, name):

    '''Flag for termination'''
    done = False

    '''Creating logger object'''
    Log_Format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = "logfile_"+str(name) +".txt", filemode = "w",format = Log_Format,
                        level = logging.INFO)
    logger = logging.getLogger()

    '''Creating plotting and display objects'''
    plot = Generate_plots(name)
    display = Display()
    observation = env.re_initialise
    display.set_visible(env.loc.copy(), env.actor_position, [])

    score = 0
    counter = 0
    while not done:
        counter +=1 # To determine to total steps taken by the agent

        '''Obtain actions from the networks'''
        action, acts = agent.greedy_epsilon(observation)

        '''Make observations'''
        observation_, reward, done, info = env.step_test(action, score)
        
        '''Update the counters and store transitions for PER'''
        steps = len(env.actorpath)
        score += reward
        agent.store_transition(observation, observation_, reward, action, int(done))
        loss = agent.learn()

        '''Visualise the agent path'''
        observation = observation_
        display.step(env.obs2D.copy(), env.actor_position, env.actorpath, acts.data.cpu().numpy(), action,
                      score, reward, env.steps,  env.walls, env.visits, env.obs2D,name)

        '''Generate Logs'''
        logger.info(f' Score {score}, Total Steps {counter},\n \t\t Counters: Path Len {steps} : Stayed {env.n_stay} : Walls {env.walls} : Visited {env.visits}, \n \t\t Actor path:  {env.actorpath} ')
        print(f'score {score}, Total Steps {counter}')
        print(f'     Path Len {steps} : Stayed {env.n_stay} : Walls {env.walls} : Visited {env.visits}')
