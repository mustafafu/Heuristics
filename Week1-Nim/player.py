import numpy as np
import game_object


BASE_MAX = 3


class Player():
    def __init__(self,player_id,num_stones,current_max):
        self.num_stones = num_stones
        self.id = player_id
        self.current_max = current_max
        self.sup_max_remove = self.compute_sup_max()
        self.state_value = -1*np.ones([self.num_stones+1,self.sup_max_remove+1]) #stones,current_max
        self.state_value[0, :] = 1
        self.state_value[:,:BASE_MAX] = 1


    def compute_sup_max(self):
        ii = self.num_stones
        MS = max(self.current_max,BASE_MAX)
        while ii>=(MS+1):
            ii -= (MS + 1)
            MS = max(MS+1,BASE_MAX)
        print('The absolute max is {}'.format(MS))
        return MS


    def get_value(self,num_stones,current_max):
        if self.state_value[num_stones,current_max] == -1:
            max_value = 0
            for i in range(1,current_max+2):
                remaining_stones = num_stones - i
                if remaining_stones < 0:
                    continue
                else:
                    max_value = max(max_value,self.get_value(remaining_stones,max(i,current_max)))
            if max_value == 1:
                self.state_value[num_stones,current_max] = 0
            else:
                self.state_value[num_stones,current_max] = 1
            return self.state_value[num_stones,current_max]
        else:
            return self.state_value[num_stones,current_max]


    def compute_next_move(self,turn_num_stones,turn_current_max):
        self.current_max = turn_current_max
        self.num_stones = turn_num_stones
        best_move = 1
        best_value = -1
        for i in range(1, self.current_max + 2):
            remaining_stones = self.num_stones - i
            if remaining_stones == 0:
                best_move = i
                return best_move
            else:
                if best_value <= self.get_value(remaining_stones, max(i, self.current_max)) :
                    best_value = self.get_value(remaining_stones, max(i, self.current_max))
                    best_move = i
        return best_move




#
# player_1 = Player('1',100,3)