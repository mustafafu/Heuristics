from reseting_game_object import Game
from reseting_player import Player
import numpy as np

BASE_MAX = 2

## Setup for the play
init_num_stones = np.random.randint(25,150)
the_game = Game(init_num_stones,BASE_MAX)
player_1 = Player('1',the_game.num_stones,the_game.current_max)
player_2 = Player('2',the_game.num_stones,the_game.current_max)

while the_game.num_stones>0:
    (whos_turn, num_stones, current_max, player_1_resets, player_2_resets, is_reset_imposed) = the_game.get_state()
    print('Player_1 has {}, Player_2 has {} remaining resets'.format(player_1_resets, player_2_resets))
    print('Its player {} s turn'.format(whos_turn))
    print('This turn reset status : {}'.format(is_reset_imposed))
    print('Remaining Stones : {}, Current max : {}'.format(num_stones,current_max))



    if whos_turn == '1':
        state = [num_stones, current_max, player_1_resets, player_2_resets, is_reset_imposed]
        best_action_value = player_1.get_best_action(state)
        action = best_action_value[0]
        the_game.take_action('1',action)
        print('Player 1 removed {} stones, is reset imposed : {}'.format(action[0],action[1]))
    else:
        state = [num_stones, current_max, player_2_resets, player_1_resets, is_reset_imposed]
        best_action_value = player_2.get_best_action(state)
        action = best_action_value[0]
        the_game.take_action('2', action)
        print('Player 2 removed {} stones, is reset imposed : {}'.format(action[0], action[1]))

        # intended_remove_number = int(input("How many stones would you like to remove?:"))
        # is_reset_imposing = int(input("Would you like to reset (1 for yes, 0 for no)?:"))
        # the_game.take_action('2', [intended_remove_number,is_reset_imposing])
        # print('You removed {} stones, is reset imposed : {}'.format(intended_remove_number,is_reset_imposing))



print('Game over player {} lost'.format(the_game.get_state()[0]))
