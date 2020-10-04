import numpy as np
import time


def get_a_busy_board(init_board,k_max, board_weight, board_cm_pos,ls_loc,rs_loc, iw_pos):
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
        return -999999
    else:
        return (cm-ls_loc) * 100




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
        #remove the smalles possible weight
        remove_idx = np.argmin(next_actions[:,1])
        random_action = next_actions[remove_idx,:]
        deepest_value = evaluate_board(board, board_weight, board_cm_pos,ls_loc,rs_loc)
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


board = get_a_busy_board(board,25, board_weight, board_cm_pos,ls_loc,rs_loc, iw_pos)
print(get_cm(board,board_weight,board_cm_pos))




player = True

while True:
    t0 = time.time()
    if np.where(board[:,1]>0)[0].shape[0] > 30:
        value, action = alphabeta(board, 2, -1_000_000, 1_000_000, True)
    elif np.where(board[:,1]>0)[0].shape[0] > 20:
        value, action = alphabeta(board, 4, -1_000_000, 1_000_000, True)
    else:
        value,action = alphabeta(board,7,-1_000_000,1_000_000,True)
    print('Number of remaining weights {}, Took {} seconds, playing action {}'.format(np.where(board[:,1]>0)[0].shape[0],time.time()-t0,action))
    board[action[0],1]=0
    player = not player
    if get_cm(board, board_weight, board_cm_pos) < ls_loc or get_cm(board, board_weight, board_cm_pos) > rs_loc:
        print('Player {} lost'.format(player))
        break




# player = True
# while 1:
#     value,action = get_best_action(board, board_weight, ls_loc, rs_loc, board_cm_pos)
#     if len(action)==0:
#         print('Player {} lost'.format(player))
#         break
#     board[action[0],1]=0
#     print(player)
#     print(get_cm(board,board_weight,board_cm_pos))
#     player = not player
#


