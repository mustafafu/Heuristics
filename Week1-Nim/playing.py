from game_object import Game
from player import Player


## Setup for the play
the_game = Game(999,3)
player_1 = Player('1',the_game.num_stones,the_game.current_max)
player_2 = Player('2',the_game.num_stones,the_game.current_max)