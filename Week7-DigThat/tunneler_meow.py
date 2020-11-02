import sys
import getopt
import math
import numpy as np

num_grid = 22
tunnel_length = 47

def build_tunnel(num_grid, tunnel_length, f):
    start_j = 2 #math.ceil(num_grid / 2) - 1
    # start_j = np.random.randint(1,num_grid+1)
    start_intersection = [1, start_j]
    print(start_intersection)
    U = num_grid - 1
    D = 0
    L = 0
    R = 0
    rem_steps = tunnel_length - (U+D)
    tunnel = ['U' for _ in range(U)]
    while rem_steps >= 1:
        # print(tunnel)
        h_to_change = np.random.randint(1,len(tunnel)-1)
        if tunnel[h_to_change-1] == 'U' and tunnel[h_to_change] == 'U':
            how_many = np.random.randint(0,1)
            rem_steps -= how_many+1
            if np.random.rand() >= 0.15:
                for i in range(how_many+1):
                    tunnel.insert(h_to_change,'R')
            else:
                for i in range(how_many+1):
                    tunnel.insert(h_to_change,'L')
        else:
            done = False
            for i in range(1,len(tunnel)-2):
                done = done or (tunnel[i] == 'U' and tunnel[i+1] == 'U')
            if not done:
                rem_steps = 0
    if len(tunnel) < tunnel_length-1:
        if tunnel[0] == 'U':
            tunnel.insert(0,'R')
        if tunnel[-1] == 'U':
            tunnel.append('L')
    tunnel = tunnel[:max(len(tunnel),tunnel_length)]
    print(tunnel)
    current_i = start_intersection
    edges = []
    for direction in tunnel:
        if current_i[0] == num_grid:
            if current_i[1] > 1:
                next_i = [current_i[0], current_i[1] - 1]
                edges.append('{},{} {},{}'.format(current_i[0], current_i[1], next_i[0], next_i[1]))
            break
        if direction == 'U':
            next_i = [current_i[0] + 1, current_i[1] ]
        elif direction == 'R':
            next_i = [current_i[0], current_i[1] + 1]
        else:
            next_i = [current_i[0], current_i[1] - 1]
        if (1<= next_i[1] <= num_grid):
            edges.append('{},{} {},{}'.format(current_i[0],current_i[1],next_i[0],next_i[1]))
            print(direction, current_i, next_i)
            current_i = next_i
        else:
            # print('I am here')
            next_i = [current_i[0] + 1, current_i[1]]
            edges.append('{},{} {},{}'.format(current_i[0],current_i[1],next_i[0],next_i[1]))
            current_i = next_i



    for edge in edges:
        f.write(edge)
        f.write('\n')



if __name__ == "__main__":
    optlist, args = getopt.getopt(sys.argv[1:], 'n:p:k:', [
        'grid=', 'phase=', 'tunnel='])
    num_grid, num_phase, tunnel_length = 0, 0, 0
    for o, a in optlist:
        if o in ("-n", "--grid"):
            num_grid = int(a)
        elif o in ("-p", "--phase"):
            num_phase = int(a)
        elif o in ("-k", "--tunnel"):
            tunnel_length = int(a)
        else:
            assert False, "unhandled option"

    f = open("tunnel", "w")
    build_tunnel(num_grid, tunnel_length, f)
    f.close()
