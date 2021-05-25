import Environment
import numpy as np
import math
import time
import multiprocessing
import Agent
import copy

class PIBB:

    def __init__(self,var=1.0,decay = 0.99, lr = 0.02,  rollouts = 8, max_it = 200, agent = None):
        self.rollouts            = rollouts         # number of games pr network
        self.variance            = var     # variance for parameters
        self.itteration          = 0         # number of itterations
        self.workers             = 4         # cores
        self.decay               = decay     # Exploration decay constant
        self.best_max_fitness    = -10000
        self.best_avg_fitness    = -10000
        self.best_min_fitness    = -10000
        self.max_iterations      = max_it
        self.lr                  = lr

        self.agent               = agent
        self.n_weights           = self.agent.get_n_weights()


        self.outfile_path        = "it_"+str(max_it)+"_var_"+str(var)+"_d_"+str(decay)+"_lr_"+str(lr)+".txt"
        f = open(self.outfile_path, "w+")
        f.close()



    def train(self, get_state, reward_func, get_move_from_action, runs_pr_rollout = 150):
        # save weights from network
        wp = self.agent.get_weights()

        print("training: "+ self.outfile_path[:-4])

        for i in range(self.max_iterations):
            print("Iteration ", i)
            #create container for rewards and epsilon_noises
            epsilons    = np.random.normal(0, self.variance, (self.rollouts, self.n_weights))

            args = []
            for rollout in range(self.rollouts):
                self.agent.set_weights(wp + epsilons[rollout])
                args.append((copy.deepcopy(self.agent), get_state, reward_func, get_move_from_action, runs_pr_rollout))

            with multiprocessing.Pool(processes=self.rollouts) as pool:
                results = np.transpose(np.asarray(pool.starmap(Environment.run, args)))

            rewards = results[0,:]
            winrates = results[1,:]
            fitness = results[2,:]
            rewards = fitness


            '''
            # preform rollouts and save Rewards
            for rollout in range(self.rollouts):
                print("Rollout ", rollout)

                # execute policy with noisy weights
                self.agent.set_weights(wp+epsilons[rollout])

                rewards[rollout] , winrates[rollout] = Environment.run(self.agent, get_state, reward_func, get_move_from_action, n=runs_pr_rollout)
            '''

            print("Avg reward: ", np.average(rewards), "Rewards", rewards)
            print("Avg winRate: ", np.average(winrates), "winRates", winrates)

            with open(self.outfile_path, "a") as textfile:
                for datapoint in rewards:
                    textfile.write(str(datapoint)+",")
                for datapoint in winrates:
                    textfile.write(str(datapoint) + ",")
                textfile.write("\n")

            # Calculate probability
            Rmin, Rmax = np.min(rewards), np.max(rewards)
            S = np.exp( self.lr * ((rewards-Rmin)/(Rmax-Rmin)))

            P = S/np.sum(S)

            # calculate change in weight
            delta_wp = np.sum(epsilons * P[:, np.newaxis], axis=0)
            wp = wp + delta_wp

            # Decay variance for future iterations
            self.variance *= self.decay
