import numpy as np
import time
import os

BASE_MAX = 2
MAX_RESET = 4


# state value is V(i,j,k,l,r) when player is playing
# i: remaining stones
# j: maximum stone
# k: remaining reset options of player 1
# l: remaining reset options of player 2
# r: 1 if reset is imposed for this turn, 0 if not

class Player():
    def __init__(self, player_id, num_stones, current_max):
        self.num_stones = num_stones
        self.id = player_id
        self.current_max = current_max
        self.sup_max_remove = self.compute_sup_max()
        save_string = os.path.join(os.getcwd(),'dp_table.npy')
        if not os.path.exists(save_string):
            print('Solving the DP-table for the first time')
            self.state_value = -1 * np.ones(
                [self.num_stones + 1, self.sup_max_remove + 1, MAX_RESET + 1, MAX_RESET + 1, 2])  # stones,current_max
            state = [self.num_stones, BASE_MAX, 4, 4, 0]
            t0 = time.time()
            best_action = self.get_best_action(state)
            print('Elapsed time : {}'.format(time.time() - t0))
            np.save(save_string,self.state_value)
        else:
            print('Loading the DP-table')
            self.state_value = np.load(save_string)


    def compute_sup_max(self):
        ii = self.num_stones
        MS = max(self.current_max, BASE_MAX)
        while ii >= (MS + 1):
            ii -= (MS + 1)
            MS = max(MS + 1, BASE_MAX)
        print('The absolute max is {}'.format(MS))
        return MS

    def get_possible_actions(self, state):
        remaining_stones = state[0]
        max_limit = state[1]
        this_remaining_resets = state[2]
        is_reset_imposed = state[4]
        if is_reset_imposed:
            range_end = min(remaining_stones, 3) + 1
            possible_removing_counts = np.arange(1, range_end)
            reset_array = np.zeros(possible_removing_counts.shape)
            if this_remaining_resets == 0:
                A = np.stack((possible_removing_counts, reset_array), axis=1)
                return A
            else:
                A_up = np.stack((possible_removing_counts, reset_array), axis=1)
                impose_reset_array = np.ones(possible_removing_counts.shape)
                A_down = np.stack((possible_removing_counts, impose_reset_array), axis=1)
                A = np.vstack((A_up, A_down))
                return A
        else:
            range_end = min(remaining_stones, max_limit + 1) + 1
            possible_removing_counts = np.arange(1, range_end)
            reset_array = np.zeros(possible_removing_counts.shape)
            if this_remaining_resets == 0:
                A = np.stack((possible_removing_counts, reset_array), axis=1)
                return A
            else:
                A_up = np.stack((possible_removing_counts, reset_array), axis=1)
                impose_reset_array = np.ones(possible_removing_counts.shape)
                A_down = np.stack((possible_removing_counts, impose_reset_array), axis=1)
                A = np.vstack((A_up, A_down))
                return A

    def get_best_action(self, state):
        A = self.get_possible_actions(state)
        A_value = np.zeros(A.shape[0])
        for ii, action in enumerate(A):
            next_player_state = [state[0] - action[0], max(state[1], action[0]), state[3], state[2] - action[1],
                                 action[1]]
            # print(next_player_state)
            if next_player_state[0] == 0:
                return action,1
            next_player_actions = self.get_possible_actions(next_player_state)
            optimal_next_player_move = []
            optimal_next_player_result = 999999
            my_next_state = []
            for next_player_action in next_player_actions:
                my_future_state = [next_player_state[0] - next_player_action[0],
                                   max(next_player_state[1], next_player_action[0]),
                                   next_player_state[3], next_player_state[2] - next_player_action[1],
                                   next_player_action[1]]
                my_future_value = self.get_value(my_future_state)
                # print('possible_states and values :{}--->{}'.format(my_future_state,my_future_value))
                if my_future_value < optimal_next_player_result:
                    optimal_next_player_result = my_future_value
                    optimal_next_player_move = next_player_action
                    my_next_state = my_future_state
            A_value[ii] = self.get_value(my_next_state)
        best_action_idx = np.argmax(A_value)
        best_action_value = np.max(A_value)
        best_action = A[best_action_idx, :]
        # print('For state : {} \n Best Action: {}, Best Return: {}'.format(state,best_action,best_action_value))
        return best_action,best_action_value

    def get_value(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        r = int(state[4])
        # print(state)
        if self.state_value[i, j, k, l, r] == -1:
            if i == 0:
                self.state_value[i, j, k, l, r] = 0
                return self.state_value[i, j, k, l, r]
            # do some computation
            [best_action,best_value]=self.get_best_action(state)
            self.state_value[i, j, k, l, r] = best_value
            return self.state_value[i, j, k, l, r]
        else:
            return self.state_value[i, j, k, l, r]



if __name__ == '__main__':
    total_stones = 1000
    player_1 = Player('1', total_stones, BASE_MAX)
    state = [total_stones, BASE_MAX, 4, 4, 0]
    t0 = time.time()
    best_action = player_1.get_best_action(state)
    print('Elapsed time : {}'.format(time.time()-t0))