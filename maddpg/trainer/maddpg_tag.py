import numpy as np
from numpy import linalg as LA
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        p_input = obs_ph_n[p_index]

        print('-'*50, int(act_pdtype_n[p_index].param_shape()[0]))
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        print(act_input_n)
        act_input_n[p_index] = act_pd.sample()
        print(act_input_n)
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


######################################################## Code for Computing SAM-NonUniform values
def a_2_pa(original_action):
    """
    transform actions output by the policy, shape=(batch_size, 5), to 
    actions in the 2d space, shape=(batch_size, 2)
    """
    physical_action = np.zeros((original_action.shape[0],2))
    physical_action[:,0] = original_action[:,1] - original_action[:,2]
    physical_action[:,1] = original_action[:,3] - original_action[:,4]

    return physical_action

def In_P(v1, v2):
    """
    Compute normalized inner product
    Input: v1, v2 shape=(batch_size,2)
    Output shape = (batch_size,)
    """
    N_IP = np.sum(v1*v2, axis=1)
    normalize_factor = LA.norm(v1, axis=1) * LA.norm(v2, axis=1)
    return N_IP/(normalize_factor + 1e-6)

def Check_IN(actions, relative_LM_pos):
    """
    Compute the normalized inner product between actions and relative postions to landmarks
    Input: one agent's actions (shape=batch_size,5) and its corresponding 
           relative landmark positions, shape=(batch_size, n_prey * 2)
    Output: inner product values, shape=(batch_size, n_prey)
    """
    n_LM = int(relative_LM_pos.shape[1]/2)
    check_values = np.zeros( (relative_LM_pos.shape[0], n_LM) )
    for ii in range(n_LM):
        check_values[:, ii] = In_P( a_2_pa(actions), relative_LM_pos[:, ii*2:(ii*2+2)] )
    
    return check_values


def SAM_NonUniform_potential(actions, relative_landmarks_pos):
    """
    array actions, shape=(n_agents, batch_size, dimension_action) inputs all actions
    array relative_landmarks_pos, shape=(n_agents, batch_size, n_prey*2) inputs all relative_landmarks_pos
    Return: SAM_NonUniform potentials, shape = (batch_size,)
    """
    n_agents = len(actions)
    sum_values = 0
    for ii in range(n_agents):
        cos_ = Check_IN(actions[ii], relative_landmarks_pos[ii, :, :])
        sum_values += np.arccos(cos_)
    
    SAM_NonUniform_values = np.squeeze(-sum_values)
    
    return SAM_NonUniform_values

def SAM_NonUniform_b(actions_n, pre_actions_n, all_observations, all_pre_observations, pre_terminals, gamma):
    """
    We adopt look back method (equation 2 in the paper)
    Inputs:
    array actions_n, shape=(n_agents, batch_size, dimension_action), inputs all actions
    array pre_actions_n inputs all the previous actions
    array all_observations, shape=(n_agents, batch_size, dimension_observation), inputs all observations
    array all_pre_observations all the previous observations
    discount factor gamma
    Returns: 
    SAM_NonUniform values
    """
    
    
    pre_values = SAM_NonUniform_potential(pre_actions_n, all_pre_observations[:,:,12:14])
    pre_values[pre_terminals] = 0
    current_values = SAM_NonUniform_potential(actions_n, all_observations[:,:,12:14])

    SAM_NonUniform_values = current_values - (1/gamma)*pre_values
    return SAM_NonUniform_values

########################################################

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        # self.replay_buffer.add(obs, act, rew, new_obs, float(done))
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), terminal)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        pre_obs_n = []
        pre_act_n = []
        pre_terminal_n = []

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index

        look_back = True
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        if look_back:
            for i in range(self.n):
                pre_obs, pre_act, _, _, _, pre_terminal = agents[i].replay_buffer.sample_index_pre(index)
                pre_obs_n.append(pre_obs)
                pre_act_n.append(pre_act)
                pre_terminal_n.append(pre_terminal)

        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            ############################################################# Apply SAM-NonUniform
            n_predators = len(target_act_next_n)-1
            if look_back:
                if self.agent_index == 3:
                    SAM_NonUniform_rew = 0
                else:
                    SAM_NonUniform_rew = SAM_NonUniform_b(np.array(act_n[0:n_predators]), np.array(pre_act_n[0:n_predators]), 
                        np.array(obs_n[0:n_predators]), np.array(pre_obs_n[0:n_predators]), pre_terminal_n[0], self.args.gamma)

            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            if look_back:
                target_q += rew + SAM_NonUniform_rew + self.args.gamma * (1.0 - done) * target_q_next
            else:
                target_q += rew + self.args.gamma * (1.0 - done) * target_q_next

            #############################################################

        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]