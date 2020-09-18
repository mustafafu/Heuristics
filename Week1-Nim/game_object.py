import numpy as np
import os
import pandas as pd
import time
import pickle



class Game():
    def __init__(self,num_stones,current_max):
        self.num_stones = num_stones
        self.whos_turn = True # true = 1st player, false = 2nd player
        self.current_max = max(current_max,3)

    def get_state(self):
        return self.num_stones,self.whos_turn,self.current_max

    def take_action(self,player,remove_count):
        if player != self.whos_turn:
            print('Wrong player')
            return
        elif remove_count > self.current_max+1:
            print('Can not remove {} stones, current max is {}'.format(remove_count,self.current_max))
            return
        else:
            self.num_stones = self.num_stones-remove_count
            self.current_max = max(self.current_max, remove_count)
            self.whos_turn = not self.whos_turn








# if __name__ == '__main__':
#     [path_lengths, paths] = main()

