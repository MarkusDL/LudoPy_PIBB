from Agent import Agent
from PIBB import PIBB
import numpy as np

state_size = 20
action_size = 4


# function for creating a state representation from observation of enviroment
def get_state(dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner):

    enemy_pieces_relative =np.multiply(np.asarray([enemy_pieces[i]+13*(i+1) for i in range(len(enemy_pieces)) ]), np.asarray(enemy_pieces, dtype = bool) )
    future_player_pieces = np.asarray(player_pieces)+dice

    #can be moved
    can_be_moved = np.zeros(4)
    for index in move_pieces:
        can_be_moved[index]=1

    #can send enemy home
    can_send_enemy_home = np.zeros(4)
    for i in range(len(player_pieces)):
        if np.any(enemy_pieces_relative == future_player_pieces[i]):
            can_send_enemy_home[i] = 1

    #is in danger
    is_in_danger = np.zeros(4)
    for i in range(len(player_pieces)):
        if np.any(np.abs(enemy_pieces_relative-player_pieces[i]) < 7):
            is_in_danger[i] = 1

    #will be in danger
    will_be_in_danger = np.zeros(4)
    for i in range(len(player_pieces)):
        if np.any(np.abs(enemy_pieces_relative - future_player_pieces[i]) < 7):
            will_be_in_danger[i] = 1

    #can reach safty
    can_reach_safty = np.zeros(4)
    for i in range(len(player_pieces)):
        if future_player_pieces[i] < 53 and not player_pieces[i] > 53 :
            can_reach_safty[i] = 1

    # just a placeholder to test if the code will run
    state = np.hstack([can_be_moved, can_send_enemy_home, is_in_danger, will_be_in_danger, can_reach_safty])

    return state

# function that creates a move based on what an action represent
counter = 0
def get_move_from_action(action, move_pieces):
    global counter
    #sort action value indexes descending order
    ordered_pice_index = [i[0] for i in sorted(enumerate(action), key=lambda k: k[1], reverse=True)]

    counter +=1
    #choose highest vales action that is able to be taken
    for pice_index in ordered_pice_index:
        if pice_index in move_pieces:
            pice_to_move = pice_index
            break
    else:
        pice_to_move = -1
        print("invalid move taken")

    if counter % 100 == None:
        print(action)
        print(ordered_pice_index)
        print(pice_to_move)

    return pice_to_move


# function that returns reward based on obersvation0 before move and observation1 after move
def reward_func( player_pieces0, enemy_pieces0, player_is_a_winner0, there_is_a_winner0, player_pieces1, enemy_pieces1, player_is_a_winner1, there_is_a_winner1):
    reward = 0
    if player_is_a_winner1:
        reward += 10

    #add reward for moving pices forward and penalty for moving back home
    reward += np.average(player_pieces1-player_pieces0)*0.01

    #add reward for sending enemies home
    reward += np.average(np.ravel(enemy_pieces0) - np.ravel(enemy_pieces1))*0.1


    return reward

if __name__ == '__main__':
    # create a instance of the class agent with a nn with input size state size and output size action size
    agent = Agent(state_size, action_size)

    # create a instance of the class PIBB with agent
    learner = PIBB(agent)

    # train the learner (PIBB using agent) with state representation, reward function and move_function
    learner.train(get_state, reward_func, get_move_from_action)

