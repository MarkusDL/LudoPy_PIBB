import ludopy
import numpy as np

n = 150
its = 200

with open("randomplayerdata.txt",'w') as f:
    for it in range(its):
        wins_list = [0,0,0,0,0,0,0,0]
        for rollout in range(8):
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

                    # get action from agent
                    if len(move_pieces0):
                            # else only one of he players are a agent

                            # take random action for other agents or based on a strategy
                            piece_to_move = move_pieces0[np.random.randint(0, len(move_pieces0))]
                    else:
                        piece_to_move = -1



                    dice1, move_pieces1, player_pieces1, enemy_pieces1, player_is_a_winner1, there_is_a_winner1 = g.answer_observation(piece_to_move)
                first_winner = g.first_winner_was
                if first_winner == 0:
                    wins+=1

                # add distance traveled by player 0 and average distance traveled for the enemies
                player_positions, enemy_positions =  g.get_pieces(seen_from=0)
                dist_from_start += np.sum(np.abs(player_positions))
                enemy_dist_from_start += np.sum(np.abs(enemy_positions))/4

                g.reset()
            wins_list[rollout] = wins

        print("itteration "+str(it)+" done")
        f.write(str(wins_list[0]/n)+","+str(wins_list[1]/n)+","+str(wins_list[2]/n)+","+str(wins_list[3]/n)+","+str(wins_list[4]/n)+","+str(wins_list[5]/n)+","+str(wins_list[6]/n)+","+str(wins_list[7]/n)+"\n")