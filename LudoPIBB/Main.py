from Agent import Agent
from PIBB import PIBB
import numpy as np

state_size = 10
action_size = 4


# function for creating a state representation from observation of enviroment
def get_state(dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner):

    # just a placeholder to test if the code will run
    state = np.zeros(10, dtype=np.double )
    return state


# function that creates a move based on what an action represent
def get_move_from_action(action, move_pieces):
    # so far just random
    pice_to_move = move_pieces[np.random.randint(0, len(move_pieces))]

    return pice_to_move


# function that returns reward based on obersvation0 before move and observation1 after move
def reward_func( player_pieces0, enemy_pieces0, player_is_a_winner0, there_is_a_winner0, player_pieces1, enemy_pieces1, player_is_a_winner1, there_is_a_winner1):
    reward = 1
    return reward


# create a instance of the class agent with a nn with input size state size and output size action size
agent = Agent(state_size, action_size)

# create a instance of the class PIBB with agent
learner = PIBB(agent)

# train the learner (PIBB using agent) with state representation, reward function and move_function
learner.train(get_state, reward_func, get_move_from_action)

