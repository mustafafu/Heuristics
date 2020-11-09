import socket
import sys
import numpy as np
from dating.utils import floats_to_msg2, candidate_to_msg


PORT = int(sys.argv[1])


def get_valid_prob(n):
    alpha = np.random.random(n)
    p = np.random.dirichlet(alpha)
    p = np.trunc(p*100)/100.0

    # ensure p sums to 1 after rounding
    p[-1] = 1 - np.sum(p[:-1])
    return p


def get_valid_weights(n):
    half = n//2

    a = np.zeros(n)
    a[:half] = get_valid_prob(half)
    a[half:] = -get_valid_prob(n - half)
    return np.around(a, 2)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', PORT))

num_string = sock.recv(4).decode("utf-8")
assert num_string.endswith('\n')

num_attr = int(num_string[:-1])
initial_weights = get_valid_weights(num_attr)
ideal_candidate = initial_weights > 0
anti_ideal_candidate = initial_weights <= 0

print('Ideal Candidate = {}'.format(np.array(ideal_candidate,dtype='int')))
print('Anti-Ideal Candidate = {}'.format(np.array(anti_ideal_candidate,dtype='int')))

sock.sendall(floats_to_msg2(initial_weights))

sock.sendall(candidate_to_msg(ideal_candidate))
sock.sendall(candidate_to_msg(anti_ideal_candidate))

current_weights = initial_weights.copy()
for i in range(20):
    # 7 char weights + commas + exclamation
    data = sock.recv(8*num_attr).decode("utf-8")
    print('%d: Received guess = %r' % (i, data))
    assert data[-1] == '\n'

    guess = np.array([float(_) for _ in data.split(sep='\n')[0].split(sep=',')])
    candid_weights = np.where(current_weights == guess)[0]
    to_change_count = num_attr//20
    to_change = np.random.choice(candid_weights,to_change_count)
    all_positives = np.where(current_weights > 0)[0]
    count = 0
    for a_positive in all_positives:
        limit = current_weights[a_positive] * 20 /100
        if count > len(to_change):
            break
        if current_weights[to_change[count]] > limit:
            current_weights[to_change[count]] -= limit
            current_weights[a_positive] += limit
            count += 1


    sock.send(floats_to_msg2(current_weights))

sock.close()
