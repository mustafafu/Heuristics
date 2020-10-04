import numpy as np
import time
import threading
import multiprocessing
from multiprocessing import Pool, TimeoutError


def get_cm(board, board_weight, board_cm_pos):
    return (np.sum(board[:, 0] * board[:, 1]) + board_cm_pos * board_weight) / (np.sum(board[:, 1]) + board_weight)


def get_feasible_remove_actions(board,board_weight,ls_loc,rs_loc,board_cm_pos):
    current_weight_idxs = np.where(board[:, 1] > 0)[0]
    current_weights = board[current_weight_idxs, 1]
    total_weight = np.sum(current_weights) + board_weight
    next_cm = (get_cm(board, board_weight, board_cm_pos) * total_weight - current_weights*board[current_weight_idxs,0]) / (total_weight - current_weights)
    feasible_actions = current_weight_idxs[np.where(np.logical_and(next_cm >= ls_loc, next_cm <= rs_loc))[0]]
    return board[feasible_actions, :]


def get_addition_action(this_board, my_weights_set, board_weight, ls_loc, rs_loc, board_cm_pos):
    available_positions = this_board[np.where(this_board[:,1]==0)[0],0]
    my_weights = np.array(list(my_weights_set))
    my_weights[::-1].sort()
    total_weight = np.sum(this_board[:,1]) + board_weight
    for weight in my_weights:
        next_cm = (get_cm(this_board, board_weight, board_cm_pos) * total_weight + weight * available_positions) / (total_weight+weight)
        feasible_positions = available_positions[np.where(np.logical_and(next_cm >= ls_loc, next_cm <= rs_loc))[0]]
        if len(feasible_positions) > 0 :
            return [feasible_positions[-1],weight]
    print('could not find a feasible placement')
    return [available_positions[-1],my_weights[0]]


def double_depth_minmax(board,depth,board_weight, ls_loc, rs_loc, board_cm_pos):
    next_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
    if next_actions.shape[0] == 0:
        # no feasible action, remove a random weight from board to drop the board
        idx = np.where(board[:,1]>0)[0]
        random_action = board[np.random.choice(idx, size=1, replace=True), :]
        return -999999, random_action[0]
    if depth == 0:
        feasible_weights = next_actions[:,1]
        feasible_positions = next_actions[:,0]
        current_weight_idxs = np.where(board[:, 1] > 0)[0]
        current_weights = board[current_weight_idxs, 1]
        total_weight = np.sum(current_weights) + board_weight
        # next_cm = (get_cm(board, board_weight, board_cm_pos) * total_weight - current_weights * board[
        #     current_weight_idxs, 0]) / (total_weight - current_weights)
        # remove_idx = np.where(next_cm<rs_loc)[0][-1]
        # some_action_idx = current_weight_idxs[remove_idx] # make it in a way to move the center of mass all the way to the right
        # return 0, board[some_action_idx, :]
        next_cm = (get_cm(board, board_weight, board_cm_pos) * total_weight - feasible_weights * feasible_positions) / (total_weight - feasible_weights)
        some_action_idx = np.argmin(next_cm)
        return 0, next_actions[some_action_idx, :]
    init_value = -999999
    A_value = init_value * np.ones(next_actions.shape[0])
    feasible_weights = next_actions[:, 1]
    feasible_positions = next_actions[:, 0]
    current_weight_idxs = np.where(board[:, 1] > 0)[0]
    current_weights = board[current_weight_idxs, 1]
    total_weight = np.sum(current_weights) + board_weight
    next_cm = (get_cm(board, board_weight, board_cm_pos) * total_weight - feasible_weights * feasible_positions) / (total_weight - feasible_weights)
    next_actions = next_actions[next_cm.argsort()]
    # next_actions = np.flip(next_actions)
    for ii, action in enumerate(next_actions):
        board[action[0], 1] = 0
        opponent_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
        if opponent_actions.shape[0] == 0:
            return 999999, action
        worst_value = 999999
        for jj,opponent_action in enumerate(opponent_actions):
            board[opponent_action[0], 1] = 0
            value, action_2 = double_depth_minmax(board.copy(),depth-2,board_weight, ls_loc, rs_loc, board_cm_pos)
            if value < worst_value:
                worst_value = value
            board[opponent_action[0], 1] = opponent_action[1]
        A_value[ii] = worst_value
        board[action[0], 1] = action[1]
        if A_value[ii] > 1:
            return A_value[ii], action
    good_actions_idx = np.where(A_value == np.max(A_value))[0]
    return 0, next_actions[np.random.choice(good_actions_idx, size=1, replace=True)[0], :]


def threadder(board,depth,board_weight, ls_loc, rs_loc, board_cm_pos):
    t0 = time.time()
    value, action = double_depth_minmax(board,depth,board_weight, ls_loc, rs_loc, board_cm_pos)
    print("Depth {} assigned to thread: {}, play action {} after {} seconds".format(depth, threading.current_thread().name,action,time.time()-t0))
    return action

def threaded_move_compute(board,default_compute_times,board_weight, ls_loc, rs_loc, board_cm_pos):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    t0 = time.time()
    init_depth = 0
    result = []
    remaining_weights = np.sum(board[:,1]>0)-1
    last_action_time = 0
    while (time.time()-t0) < default_compute_times[remaining_weights]:
        t_move_init = time.time()
        result.append(p.apply_async(threadder, (board.copy(), init_depth, board_weight, ls_loc, rs_loc, board_cm_pos) ))
        init_depth += 2
        try:
            best_action = [res.get(timeout=default_compute_times[remaining_weights]-(time.time()-t0)) for res in result]
            move_time = time.time() - t_move_init
            # if last_action_time * 1.1 > move_time:
            #     print('breaking for last action time')
            #     break
            last_action_time = move_time
            if move_time > 1.5:
                break
            if init_depth-2 >= np.sum(board[:,1]>0):
                break
        except TimeoutError:
            # print('Did not finish the process')
            p.terminate()
    print('Remaining weight count was {}, longest move compute took {}, total compute took {}'.format(remaining_weights,move_time,time.time()-t0))
    return best_action[-1]


def main():
    k_max = 25
    board_negat = -30
    board_posit = 30
    board_weight = 3
    board_cm_pos = 0 - board_negat
    board_len = abs(board_negat) + abs(board_posit) + 1
    board = np.zeros([board_len, 2], dtype='int')
    board[:, 0] = np.arange(0, board_len)
    ls_loc = -3 - board_negat
    rs_loc = -1 - board_negat
    initial_weight = 3
    iw_pos = -4 - board_negat
    board[iw_pos, 1] = initial_weight
    print(np.sum(board[:, 1] > 0))


    print(get_cm(board,board_weight,board_cm_pos))

    this_k = 25
    my_weights_set = set(np.arange(1,this_k))
    opponent_weight_set = set(np.arange(1,this_k))

    while True:
        placement = get_addition_action(board, my_weights_set, board_weight, ls_loc, rs_loc, board_cm_pos)
        print(placement)
        board[placement[0], 1] = placement[1]
        my_weights_set.remove(placement[1])
        print(get_cm(board, board_weight, board_cm_pos))
        placement = get_addition_action(board, opponent_weight_set, board_weight, ls_loc, rs_loc, board_cm_pos)
        print(placement)
        board[placement[0], 1] = placement[1]
        opponent_weight_set.remove(placement[1])
        print(get_cm(board, board_weight, board_cm_pos))

        if len(my_weights_set) == 0:
            break


    player = False

    while True:
        player = not player
        t0 = time.time()
        move_durations = np.ones([49])
        move_durations[0:13] = 5
        move_durations[13:18] = 22
        move_durations[18:22] = 6
        move_durations[22:23] = 12
        move_durations[23:30] = 16
        move_durations[30:] = 0.1
        if player:
            action = threaded_move_compute(board, move_durations,board_weight, ls_loc, rs_loc, board_cm_pos)
            board[action[0], 1] = 0
        else:
            action = threaded_move_compute(board, move_durations, board_weight, ls_loc, rs_loc, board_cm_pos)
            board[action[0], 1] = 0

        if get_cm(board, board_weight, board_cm_pos) < ls_loc or get_cm(board, board_weight, board_cm_pos) > rs_loc:
            print('Player {} lost'.format(player))
            break





if __name__ == '__main__':
    main()
