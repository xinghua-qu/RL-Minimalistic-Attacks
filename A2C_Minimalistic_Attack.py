from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from multiprocessing import Pool
from pybrain.optimization import GA,CMAES,HillClimber,RandomSearch
import os
from stable_baselines import DQN,PPO2,DDPG,A2C,ACER,ACKTR
import time
import cv2
from cv2 import *
from argparse import ArgumentParser
from multiprocessing import Pool
from itertools import product
from functools import partial


def main(game,method,pixels,tca,runname,run):

    def obj(variable,actions_0,obs):
        perturbation = np.zeros([1,84,84,4])
        for i in range(len(variable)//3):
            x=int(np.round(variable[3*i]*83))
            y=int(np.round(variable[3*i+1]*83))
            pixel_attack = int(np.round(variable[3*i+2]*254))
            x = np.clip(x, 0, 83)
            y = np.clip(y, 0, 83)
            pixel_attack = np.clip(pixel_attack, 0, 254)
            perturbation[:,x,y,:] = pixel_attack 
        np.clip(perturbation,0, 254)
        obs_new = obs + perturbation
        actions_new = model.action_probability(obs_new)
        fitness_value = max_max_distance(actions_new,actions_0)
        return fitness_value

    def evaluate(variable,obs):
        perturbation = np.zeros([1,84,84,4])
        for i in range(len(variable)//3):
            x=int(np.round(variable[3*i]*83))
            y=int(np.round(variable[3*i+1]*83))
            pixel_attack = int(np.round(variable[3*i+2]*254))
            x = np.clip(x, 0, 83)
            y = np.clip(y, 0, 83)
            pixel_attack = np.clip(pixel_attack, 0, 254)
            perturbation[:,x,y,:] = pixel_attack   
        np.clip(perturbation,0, 254)
        obs_new = obs + perturbation
        actions = model.action_probability(obs)
        actions_new = model.action_probability(obs_new)
        action, _states = model.predict(obs_new)
        obs_candi, rewards, dones, infos = env.step(action)
        return obs_candi, rewards, dones, infos, obs_new,actions_new,perturbation

    def minmax_distance(actions_new,actions_0):
        arg_max = np.argmax(actions_0[0])
        arg_min = np.argmin(actions_0[0])
        minmax_dist = actions_new[0][arg_min]-actions_new[0][arg_max]
        return minmax_dist
    
    def max_max_distance(actions_new,actions_0):
        arg_max = np.argmax(actions_0[0])
        a_candid = list(actions_new[0])
        a_candid.remove(a_candid[arg_max])
        maxmax_dist = np.max(a_candid) - actions_new[0][arg_max]
        return maxmax_dist
    
    def calculate_entropy(actions):
        entropy_actions = [-probs * np.log(probs)/np.log(len(actions)) for probs in actions]
        entropy = np.sum(entropy_actions)
        return entropy

    alg       = GA
    model = DQN.load("trained_agents/{}/{}NoFrameskip-v4".format(method,game))
    Episode_Reward = []
    Episode_Lenth  = []
    Attack_times   = []
    dir_name = 'results/{}/{}/{}/FSA_{}_TCA_{}'.format(runname,method, game, pixels,tca)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    atk_num = pixels
    bounds = [[0,1],[0,1],[0,1]]*atk_num
    env = make_atari_env('{}NoFrameskip-v4'.format(game), num_env=1, seed=run,wrapper_kwargs=None,start_index=0, allow_early_resets=True, start_method=None)
    env = VecFrameStack(env, n_stack=4)
    env.reset()
    model.set_env(env);
    obs = env.reset()
    x0 = [0.5,0.5,0.5]*atk_num
    atk_time = 0
    TrueS_array = []
    Delta_array = []
    CleanS_array= []
    for i in range(5000):
        actions = model.action_probability(obs)
        attack_significance = calculate_entropy(actions[0])
        CleanS_array.append((obs[0,:,:,3]).astype('uint8'))
        if attack_significance <= tca:
            atk_time = atk_time+1
            l = alg(lambda variable: obj(variable,actions,obs),x0,xBound=bounds, verbose=False)
            l.maximize = True
            l.maxEvaluations = 400
            res = l.learn()
            solution = list(res)[0]
            obs, rewards, dones, infos,obs_new,actions_new,perturbation = evaluate(solution,obs)
            obs_store = np.int_(obs_new)
            true_state = (obs_store[0,:,:,3]).astype('uint8')
            TrueS_array.append(true_state)
            Delta_array.append(perturbation[0,:,:,3].astype('uint8'))
        else:
            obs = np.int_(obs)
            true_state = (obs[0,:,:,3]).astype('uint8')
            TrueS_array.append(true_state)
            Delta_array.append(np.zeros([84,84]).astype('uint8'))
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)

        episode_infos = infos[0].get('episode')
        if episode_infos is not None:
            print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
            print("Atari Episode Length", episode_infos['l'])
            REWARD = episode_infos['r']
            Lenth = episode_infos['l']
            break
    size = (84, 84)
    video_dir ='results/{}_videos/{}/{}/FSA_{}_TCA_{}'.format(runname,method, game, pixels,tca)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    fps = 10
    out_true  = cv2.VideoWriter('{}/true_run_{}.avi'.format(video_dir, run), cv2.VideoWriter_fourcc(*'XVID'), fps, size) #*'PIM1'
    out_delta = cv2.VideoWriter('{}/delta_run_{}.avi'.format(video_dir, run), cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    out_clean = cv2.VideoWriter('{}/clean_run_{}.avi'.format(video_dir, run), cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for i in range(len(TrueS_array)):
        image_true = TrueS_array[i]
        x_true = np.repeat(image_true,3,axis=1)
        x_true = x_true.reshape(84, 84, 3)
        x_true[:,:,0] = 150*np.ones((84,84),dtype=int)
        x_true[:,:,1] = 150*np.ones((84,84),dtype=int)
        out_true.write(x_true)
        image_delta = Delta_array[i]
        x_delta  = np.repeat(image_delta ,3,axis=1)
        x_delta = x_delta.reshape(84, 84, 3)
        x_delta[:,:,0] = 150*np.ones((84,84),dtype=int)
        x_delta[:,:,1] = 150*np.ones((84,84),dtype=int)
        out_delta.write(x_delta)
        image_clean = CleanS_array[i]
        x_clean  = np.repeat(image_clean ,3,axis=1)
        x_clean = x_clean.reshape(84, 84, 3)
        x_clean[:,:,0] = 150*np.ones((84,84),dtype=int)
        x_clean[:,:,1] = 150*np.ones((84,84),dtype=int)
        out_clean.write(x_clean)
    cv2.destroyAllWindows()
    out_true.release()
    out_delta.release()
    out_clean.release()
    Episode_Reward.append(REWARD) 
    Episode_Lenth.append(Lenth)
    Attack_times.append(atk_time)
    data = np.column_stack((Episode_Reward,Attack_times,Episode_Lenth))
    np.savetxt('{}/run_{}.dat'.format(dir_name,run), data)
        
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-g', '--game',
                        help="the atari game ",
                        default='Breakout', type=str)
    parser.add_argument('-a', '--algorithm',
                        help="the algorithm for the policy training ",
                        default='dqn', type=str)  
    parser.add_argument('-n', '--pixels',
                        help="The number of pixels",
                        default=5, type=int) 
    parser.add_argument('-t', '--tca',
                        help="The TCA boundary",
                        default=0.9, type=float) 
    parser.add_argument('-r', '--runname',
                        help="The run name",
                        default='test', type=str) 
    args = parser.parse_args()
    return args.game, args.algorithm,args.pixels, args.tca, args.runname

if __name__ == '__main__':
    p = Pool(30)
    game,method,pixels,tca,runname = parse_arguments()
    X_input = list(range(1,30))
    func = partial(main,game,method,pixels,tca,runname)
    p.map(func,X_input)