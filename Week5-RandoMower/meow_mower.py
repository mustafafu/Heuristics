import numpy as np
import math
import multiprocessing
from multiprocessing import Pool
import json
import socket
import sys

DATA_SIZE = 4096


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


def get_first_value(first, second, prior, first_val, second_val, attachment_3, d, rope):
    return first_val + second_val - np.max([taken(third, [*prior, first, second], d, rope) for third in attachment_3])


def threshold_explore(prior, d, rope, cutoff_threshold):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    # t0 = time.time()
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
                                         (first_move, second_move, prior, first_val, second_val, attachment_3.copy(), d,
                                          rope)))
            moves.append((first_move, second_move))
    counter = 0
    total_value = []
    for i, (first_move, second_move) in enumerate(moves):
        total_value.append(results[counter].get())
        counter += 1
    best_idx = int(np.argmax(total_value))
    # print('Moves:{}-{}, Best_Value:{}'.format(moves[best_idx][0], moves[best_idx][1], total_value[best_idx]))
    # print('Pool took {} seconds'.format(time.time() - t0))
    p.close()
    return moves[best_idx][0], moves[best_idx][1]


def search_neighborhood(a1, a2, prior, rope, d):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    p.close()
    # t0 = time.time()
    attachment_1 = np.arange(np.max([a1 - 10, 0]), np.min([a1 + 11, 1100]), dtype='int16')
    attachment_2 = np.arange(np.max([a2 - 10, 0]), np.min([a2 + 11, 1100]), dtype='int16')
    move_value = np.zeros(shape=[attachment_1.shape[0], attachment_2.shape[0]])
    for i, first in enumerate(attachment_1):
        first_val = taken(first, prior, d, rope)
        for j, second in enumerate(attachment_2):
            second_val = taken(second, [*prior, first], d, rope)
            move_value[i, j] = first_val + second_val
            # results.append(p.apply_async(get_first_value,(first_move, second_move, prior, first_val, second_val, attachment_3.copy())))
            # moves.append((first_move, second_move))
    ind = np.unravel_index(np.argmax(move_value), move_value.shape)
    # print('Took {} Seconds ----- Move: {} - {}, Value: {}'.format(time.time() - t0, attachment_1[ind[0]],
    #                                                               attachment_2[ind[1]], move_value[ind]))
    return attachment_1[ind[0]], attachment_2[ind[1]]


def send_socket(s, data):
    s.sendall(data.encode('utf-8'))


def get_socket(s):
    while True:
        data = s.recv(DATA_SIZE).decode('utf-8')
        if data:
            return data


def get_argv(x):
    return sys.argv[sys.argv.index(x) + 1]


def last_move(prior, d, rope):
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    # t0 = time.time()
    min_val = round_10(np.min(prior)) if len(prior) > 0 else 550
    max_val = round_10(np.max(prior)) if len(prior) > 0 else 550
    attachment_1 = np.arange(np.max([min_val - 100, 0]), np.min([max_val + 110, 1100]), 1)
    results = []
    all_gains = []
    for a1 in attachment_1:
        results.append(p.apply_async(taken,(a1, prior, d,rope)))
    for i, a1 in enumerate(attachment_1):
        all_gains.append(results[i].get())
    best_idx = int(np.argmax(all_gains))
    p.close()
    return attachment_1[best_idx]

def play(d, rope, attachments_per_player, site, name, cutoff_threshold, isFirst):
    HOST = site.split(':')[0]
    PORT = int(site.split(':')[1])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    if isFirst:
        player_order=1
        send_socket(s, f'{player_order} {name}')
        move_idx = 1
        while move_idx <= attachments_per_player:
            moves = json.loads(get_socket(s))['moves']
            prior = moves
            if move_idx == 1:
                send_socket(s, str('{}').format(500))
                move_idx +=1
            elif move_idx == attachments_per_player:
                move = last_move(prior, d, rope)
                send_socket(s, str(move))
                move_idx +=1
                print('Sending last move as {}'.format(move))
            else:
                a1, a2 = threshold_explore(prior, d, rope, cutoff_threshold)
                b1, b2 = search_neighborhood(a1, a2, prior, rope, d)
                send_socket(s, str('{} {}').format(b1, b2))
                move_idx += 2
        for i in range(int(attachments_per_player / 2)):
            moves = json.loads(get_socket(s))['moves']
            prior = moves
            a1, a2 = threshold_explore(prior, d, rope, cutoff_threshold)
            b1, b2 = search_neighborhood(a1, a2, prior, rope, d)
            send_socket(s, str('{} {}').format(b1, b2))
    else:
        player_order = 2
        send_socket(s, f'{player_order} {name}')
        for i in range(int(attachments_per_player / 2)):
            moves = json.loads(get_socket(s))['moves']
            prior = moves
            a1, a2 = threshold_explore(prior, d, rope, cutoff_threshold)
            b1, b2 = search_neighborhood(a1, a2, prior, rope, d)
            send_socket(s, str('{} {}').format(b1, b2))
        move_idx = 1
        while move_idx <= attachments_per_player:
            moves = json.loads(get_socket(s))['moves']
            prior = moves
            if move_idx == 1:
                send_socket(s, str('{}').format(500))
                move_idx += 1
            elif move_idx == attachments_per_player:
                move = last_move(prior, d, rope)
                send_socket(s, str(move))
                move_idx += 1
                print('Sending last move as {}'.format(move))
            else:
                a1, a2 = threshold_explore(prior, d, rope, cutoff_threshold)
                b1, b2 = search_neighborhood(a1, a2, prior, rope, d)
                send_socket(s, str('{} {}').format(b1, b2))
                move_idx += 2



if __name__ == '__main__':
    d = float(get_argv('--dist'))
    rope = float(get_argv('--rope'))
    attachments_per_player = int(get_argv('--turns'))
    site = get_argv('--site')
    player_name = get_argv('--name')
    isFirst = True if '-f' in sys.argv else False
    play(d, rope, attachments_per_player, site, player_name, 0.85, isFirst)

