from game_object import Game
from player import Player
import numpy as np


## Setup for the play
init_num_stones = np.random.randint(25,999)
the_game = Game(init_num_stones,3)
player_1 = Player('1',the_game.num_stones,the_game.current_max)
player_2 = Player('2',the_game.num_stones,the_game.current_max)

while the_game.num_stones>0:
    turn_num_stones, turn_whos_turn, turn_current_max = the_game.get_state()
    print('Remaining Stones : {}, Current Player : {}, Current_max : {}'.format(turn_num_stones,turn_whos_turn,turn_current_max))
    if turn_whos_turn == True:
        player_1_move = player_1.compute_next_move(turn_num_stones,turn_current_max)
        the_game.take_action(True,player_1_move)
        print('Player 1 removed {} stones'.format(player_1_move))
    else:
        # player_2_move = player_2.compute_next_move(turn_num_stones,turn_current_max)
        # the_game.take_action(False,player_2_move)
        # print('Player 2 removed {} stones'.format(player_2_move))
        user_move = int(input("How many stones would you like to remove?:"))
        the_game.take_action(False, user_move)
        print('You removed {} stones'.format(user_move))



print('Game over palyer {} lost'.format(the_game.get_state()[1]))
