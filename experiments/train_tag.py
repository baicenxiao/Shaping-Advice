import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_tag import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

######################################################## Code for Computing SAM-Uniform values
def Dist(vec):
    """
    Input: 2d vector, shape=(batch_size, 2)
    Output: 1d vector, Euclidean distance, shape=(batch_size,)
    """
    N_IP = vec[:,0]**2 + vec[:,1]**2
    return np.sqrt(N_IP)


def Compute_dist(relative_pos):
    """
    transform the relative landmark position to Euclidean distance
    Input: relative landmark positions, shape = ( Batch_size, N_prey*2 ) = (1,2)
    Output: Distance values, shape = ( Batch_size, N_prey ) = (1,1) 
    """
    n_LM = int(relative_pos.shape[1]/2)
    check_values = np.zeros( (relative_pos.shape[0], n_LM) )
    for ii in range(n_LM):
        check_values[:, ii] = Dist( relative_pos[:, ii*2:(ii*2+2)] )
    return check_values

def SAM_uniform(observations):
    """
    Input: observations with shape= (N_agents, 1, 2)
    Output: SAM_uniform potential values ( shape=(1,) )
    """
    n_agents =  len(observations)
    
    Dist_list = []
    for ii in range(n_agents):
        Dist_list.append( Compute_dist(observations[ii]) )

    sum_dist = np.sum(np.concatenate(Dist_list, axis=1), axis=1)
            
    SAM_uniform_values = np.exp(-np.squeeze(sum_dist))
    
    return SAM_uniform_values
    

def SAM_uniform_f(all_observations, all_next_observations, gamma):
    """
    Inputs:
    array all_observations ( shape=(N_agents, 1, Dimention_obs) ) inputs all agents' observations
    array all_next_observations all the next observations ( shape=(N_agents, 1, Dimention_obs) )
    Returns: 
    SAM_uniform values ( shape=(1,) )
    """
    SAM_uniform_values = gamma*SAM_uniform(all_next_observations[:,:,12:14]) - \
                     SAM_uniform(all_observations[:,:,12:14])
    
    return SAM_uniform_values

########################################################




def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # print(env.discrete_action_space, env.discrete_action_input, 
        #         env.force_discrete_action, env.shared_reward)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_SAM_uniform = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_SAM_uniform = []
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # we use ensemble_SAM_uniform
        Ensemble_SAM_uniform = True
        print('Starting iterations...', 'Using SAM_uniform: ', Ensemble_SAM_uniform)
        
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            
            if Ensemble_SAM_uniform:
                SAM_uniform_rew_adv = SAM_uniform_f(np.array(obs_n[:3])[:,np.newaxis,:], 
                            np.array(new_obs_n[:3])[:,np.newaxis,:], arglist.gamma)

                SAM_uniform_rew_adv = 100*(0.01 + SAM_uniform_rew_adv)
            
            
            # collect experience
            for i, agent in enumerate(trainers):
                if i<3:
                    if Ensemble_SAM_uniform:
                        agent.experience(obs_n[i], action_n[i], rew_n[i]+SAM_uniform_rew_adv, new_obs_n[i], done_n[i], terminal)
                    else:
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                else:
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                if Ensemble_SAM_uniform:
                    episode_SAM_uniform[-1] += SAM_uniform_rew_adv
                else:
                    episode_SAM_uniform[-1] += 0
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                episode_SAM_uniform.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.3)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    print("steps: {}, episodes: {}, mean episode SAM_uniform: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_SAM_uniform[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    # print("steps: {}, episodes: {}, mean episode reward agent 1: {}, time: {}".format(
                    #     train_step, len(episode_rewards), np.mean(agent_rewards[1][-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode SAM_uniform: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_SAM_uniform[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                final_ep_SAM_uniform.append(np.mean(episode_SAM_uniform[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            # if len(episode_rewards) > arglist.num_episodes:
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                if arglist.exp_name:
                    rew_file_name = arglist.save_dir + arglist.exp_name + '_rewards.pkl'
                else:
                    rew_file_name = arglist.save_dir + '_rewards.pkl'

                os.makedirs(os.path.dirname(rew_file_name), exist_ok=True)
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)

                if arglist.exp_name:
                    agrew_file_name = arglist.save_dir + arglist.exp_name + '_agrewards.pkl'
                else:
                    agrew_file_name = arglist.save_dir + '_agrewards.pkl'
                
                os.makedirs(os.path.dirname(agrew_file_name), exist_ok=True)
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

                if arglist.exp_name:
                    SAM_uniform_file_name = arglist.save_dir + arglist.exp_name + '_SAM_uniform.pkl'
                else:
                    SAM_uniform_file_name = arglist.save_dir + '_SAM_uniform.pkl'
                
                os.makedirs(os.path.dirname(SAM_uniform_file_name), exist_ok=True)
                with open(SAM_uniform_file_name, 'wb') as fp:
                    pickle.dump(final_ep_SAM_uniform, fp)

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                # break
            if len(episode_rewards) > arglist.num_episodes:
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
