from Agent import Agent
from PIBB import PIBB
import numpy as np
import ludopy
import cv2

state_size = 52
action_size = 4

star_fields = np.array([5, 12, 18, 25, 31, 38, 44, 51])
globe_fields = np.array([1,9, 14, 22,27,40,48,51])
enemy_starts = np.array([14, 27, 48])

# function for creating a state representation from observation of enviroment
def get_state(dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner):
    enemy_pieces_relative =np.multiply(np.asarray([enemy_pieces[i]+13*(i+1) for i in range(len(enemy_pieces)) ]), np.asarray(enemy_pieces, dtype = bool) )%56
    future_player_pieces = np.asarray(player_pieces)+dice


    isInSafeZone = np.asarray(player_pieces >=53,dtype=float)
    isOnEnemyStart = [space in enemy_starts for space in player_pieces]
    isOnGlobe = [space in globe_fields for space in player_pieces]
    isOnStar = [space in star_fields for space in player_pieces]

    state = np.hstack([isInSafeZone, isOnEnemyStart, isOnStar, isOnGlobe])

    #can be moved
    can_be_moved = np.zeros(4)
    for index in move_pieces:
        can_be_moved[index]=1
    state = np.hstack([state,can_be_moved])

    #can reach star
    can_reach_star = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and future_player_pieces[i] in star_fields:
            can_reach_star[i] = 1
    state = np.hstack([state,can_reach_star])

    #can reach globe
    can_reach_globe = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and future_player_pieces[i] in globe_fields:
            can_reach_globe[i] = 1
    state = np.hstack([state, can_reach_globe])

    # can reach safety
    can_reach_saftyZone = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and future_player_pieces[i] >= 53:
            can_reach_saftyZone[i] = 1
    state = np.hstack([state, can_reach_saftyZone])

    #can reach home
    can_reach_home = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and can_reach_home[i] == 59:
            can_reach_home[i] = 1
    state = np.hstack([state, can_reach_home])

    #can send enemy home
    can_send_enemy_home = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and np.count_nonzero(enemy_pieces_relative == future_player_pieces[i]) == 1:
            can_send_enemy_home[i] = 1
    state = np.hstack([state, can_send_enemy_home])

    #can die from move
    can_die_from_move = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and np.count_nonzero(enemy_pieces_relative == future_player_pieces[i] ) > 1:
            can_die_from_move[i] = 1
    state = np.hstack([state, can_die_from_move])

    #can get out of home
    can_get_out_of_home = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and player_pieces[i] == 0:
            can_get_out_of_home[i] = 1
    state = np.hstack([state, can_get_out_of_home])

    #can block enemy start
    can_block_enemy_start = np.zeros(4)
    for i in range(len(player_pieces)):
        if can_be_moved[i] == 1 and future_player_pieces[i] in enemy_starts:
            can_block_enemy_start[i] = 1
    state = np.hstack([state, can_block_enemy_start])

    return state

# function that creates a move based on what an action represent

def get_move_from_action(action, move_pieces):
    #sort action value indexes descending order
    ordered_pice_index = [i[0] for i in sorted(enumerate(action), key=lambda k: k[1], reverse=True)]
    #choose highest vales action that is able to be taken
    for pice_index in ordered_pice_index:
        if pice_index in move_pieces:
            pice_to_move = pice_index
            break
    else:
        pice_to_move = -1
        print("invalid move taken")

    return pice_to_move


# function that returns reward based on obersvation0 before move and observation1 after move
def reward_func( player_pieces0, enemy_pieces0, player_is_a_winner0, there_is_a_winner0, player_pieces1, enemy_pieces1, player_is_a_winner1, there_is_a_winner1):
    reward = 0
    if player_is_a_winner1:
        reward += 1

    return reward

if __name__ == '__main__':
    for lr in [2.0, 5.0, 10.0]:
        for var in [0.4,0.2,0.1,0.05]:
            # create a instance of the class agent with a nn with input size state size and output size action size
            agent = Agent(state_size, action_size)

            # create a instance of the class PIBB with agent


            learner = PIBB(agent = agent, var = var, lr = lr)

            # train the learner (PIBB using agent) with state representation, reward function and move_function
            learner.train(get_state, reward_func, get_move_from_action)

