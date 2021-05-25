import ludopy
import numpy as np

def run(agent,get_state , reward_func, get_move_from_action, n=10):
    g = ludopy.Game()
    R = 0
    wins = 0
    dist_from_start = 0
    enemy_dist_from_start = 0

    for i in range(n):
        there_is_a_winner1 = False
        while not there_is_a_winner1:
            (dice0, move_pieces0, player_pieces0, enemy_pieces0, player_is_a_winner0, there_is_a_winner0), player_i = g.get_observation()
            # create state from observation
            state = get_state(dice0, move_pieces0, player_pieces0, enemy_pieces0, player_is_a_winner0, there_is_a_winner0)

            # get action from agent
            if len(move_pieces0):
                    # else only one of he players are a agent
                    if player_i == 0:
                        #  get action from state for agent = player 1
                        action = agent.get_action(state)
                        piece_to_move = get_move_from_action(action, move_pieces0)
                    else:
                        # take random action for other agents or based on a strategy
                        piece_to_move = move_pieces0[np.random.randint(0, len(move_pieces0))]
            else:
                piece_to_move = -1



            dice1, move_pieces1, player_pieces1, enemy_pieces1, player_is_a_winner1, there_is_a_winner1 = g.answer_observation(piece_to_move)

            if player_i == 0:
                R += reward_func( player_pieces0, enemy_pieces0, player_is_a_winner0, there_is_a_winner0, player_pieces1, enemy_pieces1, player_is_a_winner1, there_is_a_winner1)


        first_winner = g.first_winner_was
        if first_winner == 0:
            wins+=1

        # add distance traveled by player 0 and average distance traveled for the enemies
        player_positions, enemy_positions =  g.get_pieces(seen_from=0)
        dist_from_start += np.sum(np.abs(player_positions))
        enemy_dist_from_start += np.sum(np.abs(enemy_positions))/4

        g.reset()

    fitness = (wins+(dist_from_start - enemy_dist_from_start)/1000)/n
    return R/n, wins/n, fitness*300