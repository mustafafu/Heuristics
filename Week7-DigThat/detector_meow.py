import getopt
import json
import socket
import sys
import numpy as np

PLAYER_NAME = 'meow'
DATA_SIZE = 4096 * 2


def establish_connection(port):
    HOST = 'localhost'
    PORT = port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    return s


def receive_data(conn):
    while True:
        data = conn.recv(DATA_SIZE).decode()
        if data:
            return json.loads(data)


def send_data(conn, data):
    conn.sendall(json.dumps(data).encode())


def exhaustive_probe(n):
    probes = []
    for i in range(1, n + 1, 2):
        for j in range(1, n + 1, 2):
            probes.append([i, j])
    for i in range(2, n + 1, 4):
        for j in range(2, n + 1, 4):
            probes.append([i, j])
    for i in range(4,n+1,4):
        for j in range(4,n+1,4):
            probes.append([i,j])
    if n % 2 == 0:
        i = n
        for j in range(2,n+1,2):
            if [i,j] not in probes:
                probes.append([i,j])
    return probes


def get_centers(n):
    centers = []
    for i in range(2, n + 1, 4):
        for j in range(4, n + 1, 4):
            centers.append([i, j])
    for i in range(4, n + 1, 4):
        for j in range(2, n + 1, 4):
            centers.append([i, j])
    return centers


def discover_edge_parser(list_dictionary):
    this_turn_discovered = []
    for dictionary in list_dictionary:
        key = list(dictionary.keys())[0]
        edges_found = dictionary[key]
        probe_loc = [int(_) for _ in
                     list(list(dictionary.keys())[0].split(sep='[')[1].split(sep=']')[0].split(sep=','))]
        for edge_dest in edges_found:
            pair = [probe_loc, edge_dest]
            if probe_loc[0] < edge_dest[0]:
                pair = [probe_loc, edge_dest]
            if probe_loc[1] < edge_dest[1]:
                pair = [probe_loc, edge_dest]
            if probe_loc[0] > edge_dest[0]:
                pair = [edge_dest, probe_loc]
            if probe_loc[1] > edge_dest[1]:
                pair = [edge_dest, probe_loc]
            this_turn_discovered.append(pair)
    return this_turn_discovered


def find_edges(intersection, discovered_edges):
    edges = [0 for i in range(4)]
    if [intersection, [intersection[0], intersection[1] + 1]] in discovered_edges:
        edges[0] = 1
    if [intersection, [intersection[0] + 1, intersection[1]]] in discovered_edges:
        edges[1] = 1
    if [[intersection[0], intersection[1] - 1], intersection] in discovered_edges:
        edges[2] = 1
    if [[intersection[0] - 1, intersection[1]], intersection] in discovered_edges:
        edges[3] = 1
    return edges


def fill_center(center,discovered_edges,num_grid):
    edge_to_add = []
    i = center[0]
    j = center[1]
    if i==2:
        top_intersection = [i+1,j]
        top_edges = find_edges(top_intersection,discovered_edges)
        left_intersection = [i,j-1]
        left_edges = find_edges(left_intersection,discovered_edges)
        ts = np.sum(top_edges)
        ls = np.sum(left_edges)
        rs = 0
        if j+1 <= num_grid:
            right_intersection = [i,j+1]
            right_edges = find_edges(right_intersection,discovered_edges)
            rs = np.sum(right_edges)

        if rs == 1:
            edge_to_add.append([center, [i, j + 1]])
        if ls == 1:
            edge_to_add.append([[i, j - 1], center])
        if ts == 1:
            edge_to_add.append([center, [i + 1, j]])
        if (rs + ls + ts) % 2 == 1:
            edge_to_add.append([[i - 1, j], center])

    elif i == num_grid:
        a_dummy = 3
        # bottom_intersection = [i - 1, j]
        # bottom_edges = find_edges(bottom_intersection, discovered_edges)
        # left_intersection = [i, j - 1]
        # left_edges = find_edges(left_intersection, discovered_edges)
        # bs = np.sum(bottom_edges)
        # ls = np.sum(left_edges)
        # rs = 0
        # if j + 1 <= num_grid:
        #     right_intersection = [i, j + 1]
        #     right_edges = find_edges(right_intersection, discovered_edges)
        #     rs = np.sum(right_edges)
        # if rs == 1:
        #     edge_to_add.append([center, [i, j + 1]])
        # elif ls == 1:
        #     edge_to_add.append([[i, j - 1], center])
        # elif bs == 1:
        #     edge_to_add.append([[i - 1, j],center])
        # else:
        #     print('something wrong they are odd but non is one')

    elif i == num_grid-1:
        bottom_intersection = [i - 1, j]
        bottom_edges = find_edges(bottom_intersection, discovered_edges)
        left_intersection = [i, j - 1]
        left_edges = find_edges(left_intersection, discovered_edges)
        bs = np.sum(bottom_edges)
        ls = np.sum(left_edges)
        rs = 0
        if j + 1 <= num_grid:
            right_intersection = [i, j + 1]
            right_edges = find_edges(right_intersection, discovered_edges)
            rs = np.sum(right_edges)


        if rs == 1:
            edge_to_add.append([center, [i, j + 1]])
        if ls == 1:
            edge_to_add.append([[i, j - 1], center])
        if bs == 1:
            edge_to_add.append([[i - 1, j],center])
        if (rs + ls + bs) % 2 == 1:
            edge_to_add.append([center,[i+1,j]])

    else:
        if i+1 <= num_grid:
            top_intersection = [i+1,j]
            top_edges = find_edges(top_intersection,discovered_edges)
            if np.sum(top_edges)  == 1:
                edge_to_add.append([center,top_intersection])
            if np.sum(top_edges) == 3:
                print('Somthing wrong 3 edges in an intersection {}'.format(top_intersection))
        if j+1 <= num_grid:
            right_intersection = [i,j+1]
            right_edges = find_edges(right_intersection,discovered_edges)
            if np.sum(right_edges)  == 1:
                edge_to_add.append([center,right_intersection])
            if np.sum(right_edges) == 3:
                print('Somthing wrong 3 edges in an intersection {}'.format(right_intersection))
        if j-1 >= 1:
            left_intersection = [i,j-1]
            left_edges = find_edges(left_intersection,discovered_edges)
            if np.sum(left_edges) == 1:
                edge_to_add.append([left_intersection,center])
            if np.sum(left_edges) == 3:
                print('Somthing wrong 3 edges in an intersection {}'.format(left_intersection))
        if i-1 >= 1:
            bottom_intersection = [i-1,j]
            bottom_edges = find_edges(bottom_intersection,discovered_edges)
            if np.sum(bottom_edges) == 1:
                edge_to_add.append([bottom_intersection,center])
            if np.sum(bottom_edges) == 3:
                print('Somthing wrong 3 edges in an intersection {}'.format(bottom_intersection))
    return edge_to_add


if __name__ == '__main__':
    optlist, args = getopt.getopt(sys.argv[1:], 'n:p:k:', [
        'grid=', 'phase=', 'tunnel=', 'port='])
    num_grid, num_phase, tunnel_length, port = 0, 0, 0, 8000
    for o, a in optlist:
        if o in ('-n', '--grid'):
            num_grid = int(a)
        elif o in ('-p', '--phase'):
            num_phase = int(a)
        elif o in ('-k', '--tunnel'):
            tunnel_length = int(a)
        elif o in ('--port'):
            port = int(a)
        else:
            assert False, 'unhandled option'
    s = establish_connection(port)

    try:
        # Please fill in your team name here
        send_data(s, {'player_name': PLAYER_NAME})
        res = receive_data(s)
        num_grid = res['grid']
        num_phase = res['remaining_phases']
        tunnel_length = res['tunnel_length']
        print('Num Grid : {}, Num Phase : {}, Tunnel Leng: {}'.format(num_grid, num_phase, tunnel_length))
        to_probe = exhaustive_probe(num_grid)
        payload = {'phase': 'probe', 'probes': []}
        for probe in to_probe:
            payload['probes'].append(probe)
        print("payload: {}".format(payload))
        send_data(s, payload)
        already_probed = to_probe
        res = receive_data(s)
        outcome_dict = res['result']
        discovered_edges = discover_edge_parser(outcome_dict)

        centers = get_centers(num_grid)

        for center in centers:
            new_edges = fill_center(center,discovered_edges,num_grid)
            for edge in new_edges:
                discovered_edges.append(edge)

        while True:
            payload = {'phase': 'probe', 'probes': []}
            print("payload: {}".format(payload))
            send_data(s, payload)
            res = receive_data(s)
            print(res)  # gets probing report
            if res['next_phase'] == 'guess':
                break

        payload = {'phase': 'guess', 'answer': discovered_edges}
        send_data(s, payload)
        s.close()
    except KeyboardInterrupt:
        s.close()
