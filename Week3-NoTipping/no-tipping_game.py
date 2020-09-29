import numpy as np
import time

k_max = 25
board_len = 61
ls_loc = -3
rs_loc = -1
board_weight = 3
initial_weight = 3
iw_pos = -4


# just and helper function
def pos2arr(pos):
    return pos + 30


# INITIALIZE
myWeight = np.zeros(k_max)

# CREATE INITIAL BOARD
board = np.zeros(board_len)
board[pos2arr(0)] = board_weight
board[pos2arr(iw_pos)] = initial_weight
board_empty_pos = np.ones(board_len)
board_empty_pos[pos2arr(iw_pos)] = 0


# ------- numpy to quickly compute the board torques instead of lists -------------
#Torque computation vectors
left_torque_vector = np.zeros(board_len)
max_pos = 30
left_torque_vector[:pos2arr(ls_loc)] = np.arange(-1*pos2arr(ls_loc),0)
left_torque_vector[pos2arr(ls_loc)+1:] = np.arange(pos2arr(ls_loc)+1,pos2arr(max_pos)+1)-pos2arr(ls_loc)

right_torque_vector = np.zeros(board_len)
right_torque_vector[:pos2arr(rs_loc)] = np.arange(-1*pos2arr(rs_loc),0)
right_torque_vector[pos2arr(rs_loc)+1:] = np.arange(pos2arr(rs_loc)+1,pos2arr(max_pos)+1)-pos2arr(rs_loc)

def board_torques(board, left_torque_vector, right_torque_vector):
    left_torque = np.sum(board*left_torque_vector)
    right_torque = np.sum(board*right_torque_vector)
    return left_torque, right_torque


# GET INITIAL WEIGHTS
k = 16
for i in range(1, k):
    myWeight[i] = 1







