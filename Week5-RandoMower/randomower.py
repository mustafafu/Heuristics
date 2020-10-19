import numpy as np
import math
import matplotlib.pylab as plt
import multiprocessing
from multiprocessing import Pool
import itertools
import time

def area(r1, r2, d):
    if r1 <= 0 or r2 <= 0 or r1 + r2 <= d:
        return 0
    if r1 + d <= r2:
        return math.pi * r1 ** 2
    if r2 + d <= r1:
        return math.pi * r2 ** 2
    return r1 ** 2 * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)) + \
        r2 ** 2 * math.acos((d ** 2 - r1 ** 2 + r2 ** 2) / (2 * d * r2)) - \
        ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)) ** 0.5 / 2


def circum(r11, r12, r21, r22, d):
    return area(r12, r22, d) - area(r12, r21, d) - area(r11, r22, d) + area(r11, r21, d)


def taken(attachment, prior, d, rope):
    lower = max([i for i in prior if i <= attachment] or [0])
    higher = min([i for i in prior if i >= attachment] or [rope])
    return circum(lower, attachment, rope - higher, rope - attachment, d)


def round_10(x):
    return x - x % 10


def get_first_value(first,second,prior,first_val,second_val,attachment_3):
    return first_val + second_val - np.max([taken(third, [*prior, first, second], d, rope) for third in attachment_3])


def threshold_explore(prior,d,rope,cutoff_threshold):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    t0 = time.time()
    min_val = round_10(np.min(prior)) if len(prior) > 0 else 550
    max_val = round_10(np.max(prior)) if len(prior) > 0 else 550
    attachment_1 = np.arange(np.max([min_val - 100, 0]), np.min([max_val + 110, 1100]), 10, dtype='int16')
    first_gain = np.array([taken(first, prior, d, rope) for first in attachment_1])
    max_gain = np.max(first_gain)
    gain_threshold = max_gain * cutoff_threshold
    first_move_candidates = attachment_1[np.where(first_gain > gain_threshold)[0]]
    first_move_gains = first_gain[np.where(first_gain > gain_threshold)[0]]
    best_gain = 0
    results = []
    moves = []
    for i, first_move in enumerate(first_move_candidates):
        first_val = first_move_gains[i]
        attachment_2 = np.arange(np.max([np.min([min_val - 100, first_move - 100]), 0]),
                                 np.min([np.max([max_val + 110, first_move + 110]), 1110]), 10, dtype='int16')
        second_gain = np.array([taken(second, [*prior, first_move], d, rope) for second in attachment_2])
        second_max = np.max(second_gain)
        if first_val + second_max > best_gain:
            best_gain = first_val + second_max
            gain_threshold = best_gain * cutoff_threshold
        second_move_candidates = attachment_2[np.where(first_val + second_gain > gain_threshold)[0]]
        second_move_gains = second_gain[np.where(first_val + second_gain > gain_threshold)[0]]
        for j, second_move in enumerate(second_move_candidates):
            second_val = second_move_gains[j]
            attachment_3 = np.arange(np.max([np.min([min_val - 100, first_move - 100, second_move - 100]), 0]),
                                     np.min([np.max([max_val + 110, first_move + 110, second_move + 110]), 1110]), 10,
                                     dtype='int16')
            results.append(p.apply_async(get_first_value,
                                         (first_move, second_move, prior, first_val, second_val, attachment_3.copy())))
            moves.append((first_move, second_move))
    counter = 0
    total_value = []
    for i, (first_move, second_move) in enumerate(moves):
        total_value.append(results[counter].get())
        counter += 1
    best_idx = np.argmax(total_value)
    print('Moves:{}-{}, Best_Value:{}'.format(moves[best_idx][0], moves[best_idx][1], total_value[best_idx]))
    print('Pool took {} seconds'.format(time.time() - t0))
    return moves[best_idx][0], moves[best_idx][1]


def search_neighborhood(a1,a2,prior,rope,d):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    t0 = time.time()
    attachment_1 = np.arange(np.max([a1 - 10, 0]),np.min([a1 + 11, 1100]),dtype='int16')
    attachment_2 = np.arange(np.max([a2 - 10, 0]), np.min([a2 + 11, 1100]), dtype='int16')
    move_value = np.zeros(shape=[attachment_1.shape[0], attachment_2.shape[0]])
    for i,first in enumerate(attachment_1):
        first_val = taken(first, prior, d, rope)
        for j,second in enumerate(attachment_2):
            second_val = taken(second,[*prior,first],d,rope)
            move_value[i,j]=first_val+second_val
            # results.append(p.apply_async(get_first_value,(first_move, second_move, prior, first_val, second_val, attachment_3.copy())))
            # moves.append((first_move, second_move))
    ind = np.unravel_index(np.argmax(move_value),move_value.shape)
    print('Took {} Seconds ----- Move: {} - {}, Value: {}'.format(time.time()-t0,attachment_1[ind[0]],attachment_2[ind[1]],move_value[ind]))
    return attachment_1[ind[0]],attachment_2[ind[1]]


d=1000
rope=1100
prior = [501.27]

cutoff_threshold = 0.9
counter = 0
while 1:
    counter +=2
    a1,a2 = threshold_explore(prior,d,rope,cutoff_threshold)
    # prior.append(a1)
    # prior.append(a2)
    b1,b2 = search_neighborhood(a1, a2, prior, rope, d)
    prior.append(b1)
    prior.append(b2)
    if counter%4 == 0:
        cutoff_threshold *= 0.9
        print(len(prior))
    if len(prior)>15:
        break

# tot_val,at1,at2 = take_first_move(prior,d,rope)


#
# a0 = take_second_move(prior,d,rope)
# prior.append(a0)
# a1,a2 = take_first_move(prior,d,rope)
# prior.append(a1)
# prior.append(a2)
# a3,a4 = take_first_move(prior,d,rope)



# t1 = time.time()
# moves = [move for move in itertools.product(attachment_1, attachment_2, attachment_3)]
# total_value = np.array(p.map(move_value_triple, ([prior, move, d, rope] for move in moves)))
# print('Pooling took {} seconds'.format(time.time() - t1))


# num_cpu = multiprocessing.cpu_count()
# p = Pool(num_cpu)
# total_value = p.map(move_value, ((prior,move,d,rope) for move in itertools.product(attachment_1,attachment_2) ))

# prior = []
# attachment_1 = np.arange(400,700,10,dtype='int16')
# attachment_2 = np.arange(300,800,10,dtype='int16')
# attachment_3 = np.arange(200,900,10,dtype='int16')
# first_player_value = np.zeros(shape=[attachment_1.shape[0],attachment_2.shape[0],attachment_3.shape[0]])
# for i,attachment in enumerate(attachment_1):
#     first_val = taken(attachment,prior,1000,1100)
#     # print('Attachment:{},Prior:{}, Value:{}'.format(attachment,prior,first_val))
#     for j,second in enumerate(attachment_2):
#         second_val = taken(second, [*prior,attachment], 1000, 1100)
#         # print('Attachment:{}-{}, Value:{}'.format(attachment,second, second_val))
#         for k, third in enumerate(attachment_3):
#             third_val = taken(third, [*prior, attachment,second], 1000, 1100)
#             # print('Attachment:{}-{}-{}, Value:{}'.format(attachment, second,third, first_val - second_val - third_val))
#             first_player_value[i, j, k] = first_val - second_val - third_val
#
# # plt.scatter(attachment_1,first_player_value.min(axis=1))
# plt.scatter(attachment_1,first_player_value.min(axis=2).min(axis=1))
# # ind = np.unravel_index(np.argmin(first_player_value), first_player_value.shape)
# best_move = attachment_1[np.argmax(first_player_value.min(axis=2).min(axis=1))]
#
# prior.append(best_move)
# min_val = np.min(prior)
# max_val = np.max(prior)
# attachment_1 = np.arange(np.max([min_val-100,0]),np.min([max_val+100,1100]),10,dtype='int16')
# attachment_2 = np.arange(np.max([min_val-200,0]),np.min([max_val+200,1100]),10,dtype='int16')
# attachment_3 = np.arange(np.max([min_val-300,0]),np.min([max_val+300,1100]),10,dtype='int16')
# second_player_value = np.zeros(shape=[attachment_1.shape[0],attachment_2.shape[0],attachment_3.shape[0]])
# for i,attachment in enumerate(attachment_1):
#     first_val = taken(attachment,prior,1000,1100)
#     # print('Attachment:{},Prior:{}, Value:{}'.format(attachment,prior,first_val))
#     for j,second in enumerate(attachment_2):
#         second_val = taken(second, [*prior,attachment], 1000, 1100)
#         # print('Attachment:{}-{}, Value:{}'.format(attachment,second, second_val))
#         for k, third in enumerate(attachment_3):
#             third_val = taken(third, [*prior, attachment,second], 1000, 1100)
#             # print('Attachment:{}-{}-{}, Value:{}'.format(attachment, second,third, first_val - second_val - third_val))
#             second_player_value[i, j, k] = first_val + second_val - third_val
#

#
# prior = []
# attachment_1 = np.arange(400,700,10,dtype='int16')
# attachment_2 = np.arange(300,800,10,dtype='int16')
# attachment_3 = np.arange(200,900,10,dtype='int16')
# attachment_4 = np.arange(100,1000,10,dtype='int16')
# first_player_value = np.zeros(shape=[attachment_1.shape[0],attachment_2.shape[0],attachment_3.shape[0],attachment_4.shape[0]])
# for i,attachment in enumerate(attachment_1):
#     first_val = taken(attachment,prior,1000,1100)
#     print('Attachment:{}, Value:{}'.format(attachment,first_val))
#     for j,second in enumerate(attachment_2):
#         second_val = taken(second, [*prior,attachment], 1000, 1100)
#         # print('Attachment:{}-{}, Value:{}'.format(attachment,second, second_val))
#         for k, third in enumerate(attachment_3):
#             third_val = taken(third, [*prior, attachment,second], 1000, 1100)
#             # print('Attachment:{}-{}-{}, Value:{}'.format(attachment, second,third, first_val - second_val - third_val))
#             # first_player_value[i, j, k] = first_val - second_val - third_val
#             for t, fourth in enumerate(attachment_4):
#                 fourth_val = taken(fourth, [*prior, attachment, second, third], 1000, 1100)
#                 first_player_value[i, j, k, t] = first_val - second_val - third_val + fourth_val


# plt.scatter(attachment_1,[taken(attachment,prior,1000,1100) for attachment in attachment_1])

# first_value = np.array([taken(attachment,prior,1000,1100) for attachment in attachment_1])
# plt.scatter(attachment_1,[taken(attachment,prior,1000,1100) for attachment in attachment_1])



#
#
# def get_first_value(first,second,prior,first_val,second_val,attachment_3):
#     return first_val + second_val - np.max([taken(third, [*prior, first, second], d, rope) for third in attachment_3])
#
# def take_first_move(prior,d,rope):
#     num_cpu = multiprocessing.cpu_count()
#     p = Pool(num_cpu)
#     min_val = np.min(prior) if len(prior)>0 else 550
#     max_val = np.max(prior) if len(prior)>0 else 550
#     attachment_1 = np.arange(np.max([min_val - 100, 0]), np.min([max_val + 110, 1100]), 10, dtype='int16')
#     attachment_2 = np.arange(np.max([min_val - 200, 0]), np.min([max_val + 210, 1100]), 10, dtype='int16')
#     total_value = np.zeros(shape=[attachment_1.shape[0],attachment_2.shape[0]])
#     t0 = time.time()
#     results = []
#     for i,first in enumerate(attachment_1):
#         first_val = taken(first,prior,d,rope)
#         for j,second in enumerate(attachment_2):
#             second_val = taken(second,[*prior,first],d,rope)
#             attachment_3 = np.arange(np.max([np.min([min_val - 100, first-100,second-100]), 0]), np.min([np.max([max_val+110,first+110,second+110]), 1110]), 10, dtype='int16')
#             # total_value[i,j] = first_val + second_val - np.max([taken(third, [*prior, first, second], d, rope) for third in attachment_3])
#             results.append(p.apply_async(get_first_value, (first,second,prior,first_val,second_val,attachment_3.copy())))
#     counter = 0
#     for i, first in enumerate(attachment_1):
#         for j, second in enumerate(attachment_2):
#             total_value[i, j] = results[counter].get()
#             counter += 1
#     print('For loops took {} seconds'.format(time.time()-t0))
#     ind = np.unravel_index(np.argmax(total_value), total_value.shape)
#     return total_value,attachment_1,attachment_2
#
#
# def get_second_value(first,second,prior,first_val,second_val,attachment_3):
#     return first_val - second_val - np.max([taken(third, [*prior, first, second], d, rope) for third in attachment_3])
#
#
# def take_second_move(prior,d,rope):
#     num_cpu = multiprocessing.cpu_count()
#     p = Pool(num_cpu)
#     min_val = np.min(prior) if len(prior)>0 else 550
#     max_val = np.max(prior) if len(prior)>0 else 550
#     attachment_1 = np.arange(np.max([min_val - 100, 0]), np.min([max_val + 110, 1100]), 50, dtype='int16')
#     attachment_2 = np.arange(np.max([min_val - 200, 0]), np.min([max_val + 210, 1100]), 50, dtype='int16')
#     total_value = np.zeros(shape=[attachment_1.shape[0],attachment_2.shape[0]])
#     t0 = time.time()
#     results = []
#     for i,first in enumerate(attachment_1):
#         first_val = taken(first,prior,d,rope)
#         for j,second in enumerate(attachment_2):
#             second_val = taken(second,[*prior,first],d,rope)
#             attachment_3 = np.arange(np.max([np.min([min_val - 100, first-100,second-100]), 0]), np.min([np.max([max_val+110,first+110,second+110]), 1110]), 50, dtype='int16')
#             results.append(p.apply_async(get_second_value, (first,second,prior,first_val,second_val,attachment_3.copy())))
#     counter = 0
#     for i, first in enumerate(attachment_1):
#         for j, second in enumerate(attachment_2):
#             total_value[i, j] = results[counter].get()
#             counter += 1
#     print('For loops took {} seconds'.format(time.time()-t0))
#     return attachment_1[np.argmax( total_value.min(axis=1) )]
