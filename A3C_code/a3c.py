from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import chainerrl
from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_functions
import pickle as pk
import numpy as np

import gfootball.env  as env




global_enviornment_name  = "academy_empty_goal_close"
state_space_size = 39

class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(

            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """An example of A3C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size, hidden_size=32, lstm_size=32):
        #made changes from 20 to 32 and 64

        self.pi_head = L.Linear(obs_size, hidden_size)
        print(self.pi_head)
        self.v_head = L.Linear(obs_size, hidden_size)
        print(self.v_head)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        print(self.pi_lstm)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        print(self.v_lstm)
        self.pi = policies.FCGaussianPolicy(hidden_size, action_size)
        print(self.pi)
        self.v = v_functions.FCVFunction(hidden_size)
        print(self.v)
        super().__init__(self.pi_head, self.v_head,
                         self.pi_lstm, self.v_lstm, self.pi, self.v)

    def forward(self, head, lstm, tail):
        h = F.relu(head(self.state))
        h = lstm(h)

        return tail(h)

    def pi_and_v(self, state):
        self.state = state
        pout = self.forward(self.pi_head, self.pi_lstm, self.pi)
        vout = self.forward(self.v_head, self.v_lstm, self.v)

        return pout, vout




def main():

    misc.set_random_seed(0)
    env1 = env.create_environment(env_name=global_enviornment_name, render=True, representation='simple115')
    env1 = chainerrl.wrappers.CastObservationToFloat32(env1)
    timestep_limit = 180
    obs_space = env1.observation_space
    print(obs_space)
    action_space = env1.action_space
    print(action_space)
    
    #21 is the action space for the environment.
    model = A3CFFMellowmax(state_space_size,21)

    opt = rmsprop_async.RMSpropAsync(
        lr=7e-4, eps=1e-1, alpha=0.99)
    opt.setup(model)

    opt.add_hook(chainer.optimizer.GradientClipping(40))

    agent = a3c.A3C(model, opt, t_max=5, gamma=0.99,
                    beta=1e-2)

    outdir = "" #path of the output directory to store the model. Give absolute path .
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    steps=20000*5000
    eval_n_steps= 500
    eval_n_episodes=None
    eval_interval=150
    train_max_episode_len= 250
    eval_max_episode_len = 250
    chainerrl.experiments.train_agent_with_evaluation(agent, env1, steps, eval_n_steps,eval_n_episodes, eval_interval,
                                                      outdir, train_max_episode_len,
                                                      step_offset=0,eval_max_episode_len=eval_max_episode_len,
                                                      eval_env=env1, successful_score=100, step_hooks=[],
                                                      save_best_so_far_agent=True,logger=None )

if __name__ == '__main__':
    main()
