import numpy as np
import os
import pandas as pd
import time
import pickle

BASE_MAX = 2

class Game():
    def __init__(self, num_stones, current_max):
        self.num_stones = num_stones
        self.whos_turn = '1'  # '1' first player, '2' second player
        self.current_max = max(current_max, BASE_MAX)
        self.reset = False
        self.remaining_resets = [4, 4]

    def get_state(self):
        return (self.whos_turn, self.num_stones, self.current_max, self.remaining_resets[0], self.remaining_resets[1],
                self.reset)

    def take_action(self, player, action):
        if player != self.whos_turn:
            print('Wrong player')
            return

        if action[1]:
            if self.whos_turn == '1' and self.remaining_resets[0] < 1:
                print('You dont have anymore reset action, illegal move')
            if self.whos_turn == '2' and self.remaining_resets[1] < 1:
                print('You dont have anymore reset action, illegal move')

        remove_count = action[0]
        if self.reset:
            if remove_count > 3:
                print('Reset imposed please remove 1-2-3 stones')
                return
            else:
                self.num_stones = self.num_stones - remove_count
                self.reset = action[1]
                self.current_max = max(self.current_max, remove_count)
                if self.whos_turn == '1':
                    self.whos_turn = '2'
                    self.remaining_resets[0] = self.remaining_resets[0] - action[1]
                elif self.whos_turn == '2':
                    self.whos_turn = '1'
                    self.remaining_resets[1] = self.remaining_resets[1] - action[1]
        else:
            if remove_count > self.current_max + 1:
                print('Can not remove {} stones, current max is {}'.format(remove_count, self.current_max))
                return
            else:
                self.num_stones = self.num_stones - remove_count
                self.reset = action[1]
                self.current_max = max(self.current_max, remove_count)
                if self.whos_turn == '1':
                    self.whos_turn = '2'
                    self.remaining_resets[0] = self.remaining_resets[0] - action[1]
                elif self.whos_turn == '2':
                    self.whos_turn = '1'
                    self.remaining_resets[1] = self.remaining_resets[1] - action[1]








# if __name__ == '__main__':
#     [path_lengths, paths] = main()
