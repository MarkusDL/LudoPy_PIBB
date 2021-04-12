import Environment
import numpy as np
import math

class PIBB:

    def __init__(self, agent):
        self.rollouts            = 8         # number of games pr network
        self.variance            = 0.015     # variance for parameters
        self.init_var_boost      = 2         # gain on variance in first iteration
        self.itteration          = 0         # number of itterations
        self.workers             = 4         # cores
        self.h                   = 10        # Exploration constant
        self.decay               = 0.995     # Exploration decay constant
        self.best_max_fitness    = -10000
        self.best_avg_fitness    = -10000
        self.best_min_fitness    = -10000
        self.max_iterations      = 100
        self.lr              = 0.01

        self.agent               = agent
        self.n_weights           = self.agent.get_n_weights()

    def train(self, get_state, reward_func, get_move_from_action, runs_pr_rollout =10, self_play = False ):
        # save weights from network
        wp = self.agent.get_weights()

        for i in range(self.max_iterations):
            print("Iteration ", i)
            #create container for rewards and epsilon_noises
            rewards     = np.zeros((self.rollouts), dtype=float)
            epsilons    = np.zeros((self.rollouts, self.n_weights), dtype=float)

            # preform rollouts and save Rewards and epsilon noises
            for rollout in range(self.rollouts):
                print("Rollout ", rollout)
                # generate noise for weights
                epsilon = np.random.normal(0, self.variance, self.n_weights)
                epsilons[rollout] = epsilon

                # execute policy with noisy weights
                self.agent.set_weights(wp+epsilon)

                rewards[rollout] = Environment.run(self.agent, get_state, reward_func, get_move_from_action, n=runs_pr_rollout, self_play=self_play)

            # Calculate propapility
            S = np.zeros((self.rollouts), dtype=float)
            for k in range(self.rollouts):
                # calculate Sk
                Rk, Rmin, Rmax = rewards[k], np.min(rewards), np.max(rewards)
                S[k] = math.exp( self.lr* ((Rk-Rmin)/(Rmax-Rmin)))

            P = S/np.sum(S)

            delta_wp = np.sum(epsilons * P[:, np.newaxis], axis=0)
            wp = wp + delta_wp
