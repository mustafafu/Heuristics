import numpy as np
import time
import threading
import multiprocessing
from multiprocessing import Pool, TimeoutError


def get_a_busy_board(init_board, k_max, board_weight, board_cm_pos,ls_loc,rs_loc, iw_pos):
    weights = np.hstack([np.arange(1, k_max, 1), np.arange(1, k_max, 1)])
    locations = np.delete(init_board[:, 0].copy(), iw_pos)
    board = np.zeros(init_board.shape, dtype='int')
    board[:,0] = init_board[:,0]
    board[:,1] = init_board[:,1]
    board[np.random.choice(locations, size=len(weights), replace=False), 1] = weights
    current_cm = get_cm(board, board_weight, board_cm_pos)
    while current_cm < ls_loc or current_cm > rs_loc:
        board = np.zeros(init_board.shape, dtype='int')
        board[:, 0] = init_board[:, 0]
        board[:,1] = init_board[:,1]
        board[np.random.choice(locations, size=len(weights), replace=False), 1] = weights
        current_cm = get_cm(board, board_weight, board_cm_pos)
    return board


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


def get_best_action(board, depth, board_weight, ls_loc, rs_loc, board_cm_pos):
    next_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
    if next_actions.shape[0]==0:
        return 0,[]
    if depth==0:
        idx = np.arange(0,next_actions.shape[0])
        return 0,next_actions[np.random.choice(idx,size=1 , replace=True),:]
    A_value = np.zeros(next_actions.shape[0])
    for ii,action in enumerate(next_actions):
        board[action[0], 1] = 0
        opponent_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
        if opponent_actions.shape[0] == 0:
            board[action[0], 1] = action[1]
            return 1, action
        optimal_next_player_move = []
        optimal_next_player_result = 999999
        my_future_worst_value = []
        for opponent_action in opponent_actions:
            board[opponent_action[0], 1] = 0
            my_future_value = get_value(board, depth-1, board_weight, ls_loc, rs_loc, board_cm_pos)
            if my_future_value < optimal_next_player_result:
                optimal_next_player_result = my_future_value
                optimal_next_player_move = opponent_action
                A_value[ii] = my_future_value
            board[opponent_action[0], 1] = opponent_action[1]
        board[action[0], 1] = action[1]
    best_action_idx = np.argmax(A_value)
    best_action_value = np.max(A_value)
    best_action = next_actions[best_action_idx, :]
    # print('For state : {} \n Best Action: {}, Best Return: {}'.format(state,best_action,best_action_value))
    return best_action_value, best_action


def get_value(board, board_weight, depth, ls_loc, rs_loc, board_cm_pos):
    next_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
    if next_actions.shape[0] == 0:
        return 0
    elif depth == 0:
        return 0
    else:
        best_value, best_action = get_best_action(board, depth-1, board_weight, ls_loc, rs_loc, board_cm_pos)
        return best_value


def evaluate_board(board, board_weight, board_cm_pos,ls_loc,rs_loc):
    cm = get_cm(board, board_weight, board_cm_pos)
    if cm > rs_loc or cm < ls_loc:
        print('Something wrong with alphabeta-eval board')
    return 1000 * (cm - ls_loc)
    # if isMe:
    #     return 1000 * (cm - ls_loc)
    # else:
    #     return 1000 * (cm -ls_loc)
    # if cm > rs_loc or cm < ls_loc:
    #     return -999999
    # else:
    #     return 100


def alphabeta(board,depth,alpha,beta,isMe):
    next_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
    if next_actions.shape[0] == 0:
        # no feasible action, remove a random weight from board to drop the board
        idx = np.where(board[:,1]>0)[0]
        random_action = board[np.random.choice(idx, size=1, replace=True), :]
        if isMe:
            return -999999, random_action[0]
        else:
            return 999999, random_action[0]
    if depth == 0:
        #remove the heaviest possible weight
        right_torques = next_actions[29:,0]*next_actions[29:,1]
        if len(right_torques) > 0:
            remove_idx = np.argmax(right_torques) + 29
        else:
            remove_idx = np.argmax(next_actions[:,1])
        random_action = next_actions[remove_idx,:]
        deepest_value = evaluate_board(board, board_weight, board_cm_pos,ls_loc,rs_loc,isMe)
        return deepest_value, random_action
    if isMe:
        value = -999999
        A_value = value * np.ones(next_actions.shape[0])
        for ii, action in enumerate(next_actions):
            board[action[0], 1] = 0
            opponent_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
            if opponent_actions.shape[0] == 0:
                idx = np.where(board[:, 1] > 0)[0]
                random_action = board[np.random.choice(idx, size=1, replace=True), :]
                board[action[0], 1] = action[1]
                return 999999,random_action[0]
            value = max(value, alphabeta(board, depth-1, alpha, beta, False)[0])
            A_value[ii] = value
            alpha = max(value,alpha)
            board[action[0], 1] = action[1]
            if alpha >= beta:
                break
        return value, next_actions[np.argmax(A_value),:]
    else:
        value = 999999
        A_value = value * np.ones(next_actions.shape[0])
        for ii, action in enumerate(next_actions):
            board[action[0], 1] = 0
            opponent_actions = get_feasible_remove_actions(board, board_weight, ls_loc, rs_loc, board_cm_pos)
            if opponent_actions.shape[0] == 0:
                idx = np.where(board[:, 1] > 0)[0]
                random_action = board[np.random.choice(idx, size=1, replace=True), :]
                board[action[0], 1] = action[1]
                return -999999,random_action[0]
            value = min(value, alphabeta(board, depth - 1, alpha, beta, True)[0])
            A_value[ii] = value
            beta = min(value, beta)
            board[action[0], 1] = action[1]
            if beta <= alpha:
                break
        return value, next_actions[np.argmin(A_value),:]


def threadder(board,depth,alpha,beta,isMe):
    t0 = time.time()
    value, action = alphabeta(board, depth, alpha,beta,isMe)
    print("Depth {} assigned to thread: {}, play action {} after {} seconds".format(depth, threading.current_thread().name,action,time.time()-t0))
    return action

def threaded_move_compute(board,default_compute_times):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    t0 = time.time()
    init_depth = 2
    result = []
    remaining_weights = np.sum(board[:,1]>0)
    last_action_time = 0
    while (time.time()-t0) < 10:
        t_move_init = time.time()
        result.append(p.apply_async(threadder, (board.copy(), init_depth , -1_000_000, 1_000_000, True)))
        init_depth += 1
        try:
            best_action = [res.get(timeout=10-(time.time()-t0)) for res in result]
            move_time = time.time() - t_move_init
            if last_action_time * 2 > move_time:
                print('breaking for last action time')
                break
            last_action_time = move_time
            if move_time > 1.5:
                break
        except TimeoutError:
            print('Did not finish the process')
            p.terminate()
    print(time.time() - t0)
    return best_action[-1]

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
    if player:
        action = threaded_move_compute(board, 10 * np.ones([49]))
        board[action[0], 1] = 0
    else:
        print('your turn')
        feasible_remove_actions = get_feasible_remove_actions(board,board_weight,ls_loc,rs_loc,board_cm_pos)
        print(feasible_remove_actions)
        intended_location = int(input("Where is the weight you want to remove:"))
        is_reset_imposing = int(input("What is the weight you want to remove:"))
        action = np.array([intended_location,is_reset_imposing])
        board[action[0], 1] = 0

    if get_cm(board, board_weight, board_cm_pos) < ls_loc or get_cm(board, board_weight, board_cm_pos) > rs_loc:
        print('Player {} lost'.format(player))
        break





#
# player = True
#
# while True:
#     t0 = time.time()
#     action = threaded_move_compute(board, 10*np.ones([49]))
#     print('Number of remaining weights {}, Took {} seconds, playing action {}'.format(np.where(board[:,1]>0)[0].shape[0],time.time()-t0,action))
#     board[action[0],1]=0
#     player = not player
#     if get_cm(board, board_weight, board_cm_pos) < ls_loc or get_cm(board, board_weight, board_cm_pos) > rs_loc:
#         print('Player {} lost'.format(player))
#         break



# player = True
#
# while True:
#     t0 = time.time()
#     if np.where(board[:,1]>0)[0].shape[0] > 30:
#         value, action = alphabeta(board, 3, -1_000_000, 1_000_000, True)
#     elif np.where(board[:,1]>0)[0].shape[0] > 20:
#         value, action = alphabeta(board, 4, -1_000_000, 1_000_000, True)
#     elif np.where(board[:,1]>0)[0].shape[0] > 15:
#         value,action = alphabeta(board,7,-1_000_000,1_000_000,True)
#     else:
#         value, action = alphabeta(board, 10, -1_000_000, 1_000_000, True)
#     print('Number of remaining weights {}, Took {} seconds, playing action {}'.format(np.where(board[:,1]>0)[0].shape[0],time.time()-t0,action))
#     board[action[0],1]=0
#     player = not player
#     if get_cm(board, board_weight, board_cm_pos) < ls_loc or get_cm(board, board_weight, board_cm_pos) > rs_loc:
#         print('Player {} lost'.format(player))
#         break











#
# board = get_a_busy_board(board,25, board_weight, board_cm_pos,ls_loc,rs_loc, iw_pos)
# print(get_cm(board,board_weight,board_cm_pos))
#
#
#
#




#
# multiple_results = [p.apply_async(threadder, (board.copy(), i, -1_000_000, 1_000_000, True)) for i in range(2,9)]
# try:
#     print([res.get(timeout=10) for res in multiple_results])
# except TimeoutError:
#     print('Didnt finish yet')



# t2 = threading.Thread(target=threadder, args=(board.copy(), 2, -1_000_000, 1_000_000, True))
# t3 = threading.Thread(target=threadder, args=(board.copy(), 3, -1_000_000, 1_000_000, True))
# t4 = threading.Thread(target=threadder, args=(board.copy(), 4, -1_000_000, 1_000_000, True))
# t5 = threading.Thread(target=threadder, args=(board.copy(), 5, -1_000_000, 1_000_000, True))
# t6 = threading.Thread(target=threadder, args=(board.copy(), 6, -1_000_000, 1_000_000, True))
# t7 = threading.Thread(target=threadder, args=(board.copy(), 7, -1_000_000, 1_000_000, True))
# t8 = threading.Thread(target=threadder, args=(board.copy(), 8, -1_000_000, 1_000_000, True))
# # starting thread 1
# t2.start()
# # starting thread 2
# t3.start()
# # starting thread 3
# t4.start()
# # starting thread 4
# t5.start()
# # starting thread 4
# t6.start()
# # starting thread 4
# t7.start()
# # starting thread 4
# t8.start()

