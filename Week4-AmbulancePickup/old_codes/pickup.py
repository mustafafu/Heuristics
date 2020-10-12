import numpy as np
import sys
from sklearn.cluster import KMeans
import networkx as nx
import itertools
from concurrent import futures
import time
from multiprocessing import Pool, freeze_support, cpu_count

def read_all(hospital_locations = False):
    n_patient = 0
    n_hospital = 0
    hospitals = []
    patients = []
    state = 0
    # for line in sys.stdin:
    file1 = open('data1', 'r')
    Lines = file1.readlines()
    for line in Lines:
        ln = line.split()
        if not ln:
            continue
        if state == 0:
            state = 1
        elif state == 1:
            vals = ln[0].split(sep=',')
            if vals[0].isdigit():
                # vvvvvv You may want to save these data
                ptnt_x = int(vals[0])
                ptnt_y = int(vals[1])
                ptnt_time = int(vals[2])
                n_patient +=1
                patients.append([ptnt_x,ptnt_y,ptnt_time])
            else:
                # Ignore the line with column names of the second part
                # site day beginhour endhour
                state = 2
        elif state == 2:
            if hospital_locations:
                vals = ln[0].split(sep=',')
                if vals[0].isdigit():
                    hospital_x = int(vals[0])
                    hospital_y = int(vals[1])
                    hospital_ambulance = int(vals[2])
                    hospitals.append([hospital_x,hospital_y,hospital_ambulance])
            else:
                vals = ln[0].split(sep=',')
                if vals[0].isdigit():
                    n_hospital+=1
                    hospitals.append([-1,-1,int(vals[0]) ] )
    return patients,hospitals


def compute_patient_distances(patient_array):
    patient_x_diff = np.abs(patient_array[:,0].reshape(-1,1) - patient_array[:,0].reshape(1,-1))
    patient_y_diff = np.abs(patient_array[:,1].reshape(-1,1) - patient_array[:,1].reshape(1,-1))
    return patient_x_diff + patient_y_diff


def compute_cross_distance(array_x_y):
    x_diff = np.abs(array_x_y[:,0].reshape(-1,1) - array_x_y[:,0].reshape(1,-1))
    y_diff = np.abs(array_x_y[:,1].reshape(-1,1) - array_x_y[:,1].reshape(1,-1))
    return x_diff + y_diff


def compute_hospital_patient_distances(patient_array,hospital_array):
    if len(hospital_array.shape)==1:
        patient_x_diff = np.abs(hospital_array[0].reshape(-1, 1) - patient_array[:, 0].reshape(1, -1))
        patient_y_diff = np.abs(hospital_array[1].reshape(-1, 1) - patient_array[:, 1].reshape(1, -1))
    else:
        patient_x_diff = np.abs(hospital_array[:, 0].reshape(-1, 1) - patient_array[:, 0].reshape(1, -1))
        patient_y_diff = np.abs(hospital_array[:, 1].reshape(-1, 1) - patient_array[:, 1].reshape(1, -1))
    return patient_x_diff + patient_y_diff


def compute_distance(x,distances):
    num_patients = len(x)
    dist_x = 0
    for i in range(num_patients-1):
        dist_x += distances[x[i]][x[i+1]]
    return dist_x

def find_deadline(x,patients):
    return np.min(patients[x,2])


# def feasibility_patient_array(x,distances,patients):
#     if compute_distance(x,distances) <= find_deadline(x,patients):
#         return x
#     else:
#         return

def feasibility_patient_array(x):
    global distances
    global patients
    if len(set(x)) == len(x):
        if compute_distance(x,distances) <= find_deadline(x,patients):
            # if len(np.unique(x)) == len(x):
            return x


def compute_route_distance(routes,distances):
    route_len = routes.shape[1]
    route_distances = np.zeros(routes.shape[0])
    for i in range(route_len - 1):
        route_distances += distances[routes[:,i],routes[:,i+1]]
    return route_distances


def return_shortest_route_w_tsp(res,distances,hospital_start_idxs,hospital_end_idxs,patients):
    num_patient = res.shape[1]
    list = [np.arange(0,num_patient)]*num_patient
    min_dists = 9999999*np.ones([res.shape[0]])
    best_routes = np.zeros([res.shape[0], res.shape[1] + 2], dtype='int')
    first_time = True
    for a in itertools.product(*[*list]):
        if len(a) != len(set(a)):
            continue
        else:
            tmp_res = res[:,a]
            routes = np.zeros([tmp_res.shape[0],tmp_res.shape[1]+2],dtype='int')
            closest_starting_hospitals = hospital_start_idxs[distances[hospital_start_idxs.reshape(-1,1),tmp_res[:,0]].argmin(axis=0)]
            closest_finishing_hospitals = hospital_end_idxs[distances[hospital_end_idxs.reshape(-1,1),tmp_res[:,-1]].argmin(axis=0)]
            routes[:,0] = closest_starting_hospitals
            routes[:,-1] = closest_finishing_hospitals
            routes[:,1:-1] = tmp_res
            route_distances = compute_route_distance(routes,distances)
            if first_time:
                route_deadlines = np.min(patients[routes[:,1:-1],2],axis=1)
                first_time=False
            better_routes_idxs = np.where(min_dists > route_distances)[0]
            min_dists[better_routes_idxs] = route_distances[better_routes_idxs]
            best_routes[better_routes_idxs,:] = routes[better_routes_idxs,:]
    return (best_routes[np.where(min_dists <= route_deadlines)[0]],route_deadlines[np.where(min_dists <= route_deadlines)[0]],min_dists[np.where(min_dists <= route_deadlines)[0]])



# class Scheduler():
#     def __init__(self, routes, route_deadlines, route_distances, patient_indexes):
#         self.routes = routes
#         self.chosen_routes = []
#         self.route_deadlines = route_deadlines
#         self.max_deadline = np.max(self.route_deadlines)
#         self.route_distances = route_distances
#         self.patients_set = set(patient_indexes[:])
#         self.dp_matrix = -1 * np.ones([self.max_deadline,1],dtype=np.int16)
#


    # max_time = np.max(route_deadlines) + 1
    # scheduled_patients = set()
    # greedy_scheduler_backwards(max_time, routes, route_deadlines, route_distances, scheduled_patients)

def greedy_scheduler_backwards(max_time,routes, route_deadlines, route_distances, scheduled_patients):
    remaining_route_idxs = np.where(route_deadlines <= max_time)[0]
    if len(remaining_route_idxs) == 0:
        return []
    remaining_route_idxs = remaining_route_idxs[np.argsort(route_deadlines[remaining_route_idxs] - route_distances[remaining_route_idxs])]
    remaining_route_idxs = remaining_route_idxs[::-1]
    candidate_route_counter = 0
    candidate_route = routes[remaining_route_idxs[candidate_route_counter]]
    while len(scheduled_patients.intersection(candidate_route)) > 0:
        candidate_route_counter +=1
        candidate_route = routes[remaining_route_idxs[candidate_route_counter]]
    if candidate_route_counter < len(remaining_route_idxs):
        next_max_time = route_deadlines[remaining_route_idxs[candidate_route_counter]] - route_distances[
            remaining_route_idxs[candidate_route_counter]]
        for patient in candidate_route[1:-1]:
            scheduled_patients.add(patient)
        other_route_idxs = greedy_scheduler_backwards(next_max_time, routes, route_deadlines, route_distances, scheduled_patients)
        other_route_idxs.append(candidate_route)
        return other_route_idxs
    else:
        return []

def greedy_scheduler_forwards(current_time, routes, route_deadlines, route_distances, scheduled_patients, last_hospital):
    if last_hospital is None:
        remaining_route_idxs = np.where(route_deadlines - route_distances >= current_time)[0]
    else:
        remaining_route_idxs = np.where(np.logical_and((route_deadlines - route_distances >= current_time),(routes[:, 0] == last_hospital)))[0]
    if len(remaining_route_idxs) == 0:
        return [], 0
    best_value = 0
    this_route = []
    for ii,candidate_route in enumerate(routes[remaining_route_idxs]):
        if len(scheduled_patients.intersection(candidate_route)) > 0:
            continue
        next_time = current_time + route_distances[remaining_route_idxs[ii]]
        next_scheduled_patients = set(candidate_route[1:-1]).union(scheduled_patients)
        other_routes, value_forward = greedy_scheduler_forwards(next_time, routes, route_deadlines, route_distances, next_scheduled_patients, last_hospital = candidate_route[-1])
        my_value = value_forward + len(candidate_route[1:-1])
        if my_value > best_value:
            best_value = my_value
            this_route = other_routes + [candidate_route.tolist()]
    return this_route, best_value



if __name__ == '__main__':
    patients, hospitals = read_all(hospital_locations = False)
    patients = np.array(patients)
    hospitals = np.array(hospitals)
    num_hospitals = hospitals.shape[0]
    if hospitals[0,0] == -1:
        kmeans = KMeans(n_clusters=num_hospitals).fit(patients[:,:2])
        hospital_locations = np.round(kmeans.cluster_centers_).astype('int')
        hospitals[:,:2] = hospital_locations
    else:
        hospital_locations = hospitals[:,:2]

    V_c = patients.shape[0]
    W = hospitals.shape[0]
    patient_idxs = np.arange(0,V_c)
    hospital_start_idxs = np.arange(V_c,V_c+W)
    hospital_end_idxs = np.arange(V_c+W, V_c+W+W)
    all_indexes = np.array([*patient_idxs,*hospital_start_idxs,*hospital_end_idxs])

    all_locations_matrix = np.zeros([V_c+W+W,2],dtype='int')
    all_locations_matrix[:V_c,:] = patients[:,:2]
    all_locations_matrix[V_c:V_c+W, :] = hospitals[:, :2]
    all_locations_matrix[V_c+W:V_c + W + W, :] = hospitals[:, :2]

    distances = compute_cross_distance(all_locations_matrix)
    distances += 1
    for i in all_indexes:
        distances[i][i] = 0



    # one_hop_routes = find_one_hop_routes(distances,patient_idxs,hospital_start_idxs,hospital_end_idxs,patients)
    # two_hop_routes = find_two_hop_routes(distances,patient_idxs,patients,one_hop_routes)
    # three_hop_routes = find_three_hop_routes(distances,patient_idxs,patients,two_hop_routes)


    number_of_patients = 2
    aaa = [patient_idxs] * number_of_patients
    t0=time.time()
    params = [x for x in itertools.product(*aaa) if len(x) == len(set(x))]
    print(time.time()-t0)



    t0 = time.time()
    res = np.array(params,dtype=np.int16)
    res.sort(axis=1)
    print(time.time() - t0)
    res = np.unique(res,axis=0)
    print(time.time() - t0)
    # I can precompute and save this easily.
    # hyp_patients = 250
    # res = res[np.where(res.max(axis=1) < hyp_patients)[0]]
    t0 = time.time()
    feasible_routes,feasible_deadlines,feasible_distances = return_shortest_route_w_tsp(res, distances, hospital_start_idxs, hospital_end_idxs, patients)
    print(time.time() - t0)


    feasible_patients = feasible_routes[:,1:-1]
    triplets = [(*y[0],y[1]) for y in itertools.product(feasible_patients.tolist(),patient_idxs)]
    triplets = [triplet for triplet in triplets if len(triplet)==len(set(triplet))]
    t0 = time.time()
    res = np.array(triplets, dtype=np.int16)
    res.sort(axis=1)
    print(time.time() - t0)
    res = np.unique(res, axis=0)
    print(time.time() - t0)
    # I can precompute and save this easily.
    # hyp_patients = 250
    # res = res[np.where(res.max(axis=1) < hyp_patients)[0]]
    t0 = time.time()
    routes, route_deadlines, route_distances = return_shortest_route_w_tsp(res, distances,hospital_start_idxs,
                                                                                          hospital_end_idxs, patients)
    print(time.time() - t0)


    routes[:,-1] -= hospital_end_idxs[0] - hospital_start_idxs[0]
    current_time = 0
    last_hospital = None
    scheduled_patients = set()

    one_ambulance, value = greedy_scheduler_forwards(current_time, routes, route_deadlines, route_distances, scheduled_patients, last_hospital)
    scheduled_patients = scheduled_patients.union(set(np.array(one_ambulance)[:,1:-1].ravel().tolist()))
    one_ambulance, value = greedy_scheduler_forwards(current_time, routes, route_deadlines, route_distances,
                                                     scheduled_patients, last_hospital)
    scheduled_patients = scheduled_patients.union(set(np.array(one_ambulance)[:, 1:-1].ravel().tolist()))

    # # you can use whatever, but your machine core count is usually a good choice (although maybe not the best)
    # pool = Pool(cpu_count())
    # t0 = time.time()
    # res = pool.map(feasibility_patient_array, params)
    # print(time.time() - t0)
    # t0 = time.time()
    # res = np.array([i for i in res if not (i is None)])
    # res.sort(axis=1)
    # res = np.unique(res,axis=0)
    # print(time.time()-t0)





    # total_error = 0
    #
    # with futures.ProcessPoolExecutor() as pool:
    #     results = pool.map(wrapped_some_function_call, all_args)
    #     for feasible in pool.map(feasibility_patient_array, (params)):





    #
    # def check_feasibility():
    #
    #     return
    #
    #
    #

    # patient_distances = compute_patient_distances(patients)
    # hospital_patient_distances = compute_hospital_patient_distances(patients,hospital_locations)
    # # the following is for 1 hospital special case
    # saveable_patients = patients[patients[:,2] < 2*hospital_patient_distances[0]]


# def main():
#     patients, n_ambulance = read_all()
#     return



















# def graph_functions():
#     G = nx.Graph()
#     G.add_nodes_from([(p_idx,{'x':patient[0],'y':patient[1],'ttl':patient[2],'color':1,'type':'patient'}) for
#                       p_idx,patient in enumerate(patients)])
#     G.add_nodes_from([(h_idx,{'x':hospital[0],'y':hospital[1],'num_ambulance':hospital[2],'color':2,'type':'hospital'}) for
#                       h_idx,hospital in zip(hospital_start_idxs,hospitals)])
#     G.add_nodes_from([(h_idx, {'x': hospital[0], 'y': hospital[1], 'num_ambulance': hospital[2],'color':3, 'type': 'hospital'}) for
#          h_idx, hospital in zip(hospital_end_idxs, hospitals)])
#
#     G.add_weighted_edges_from([(node_1,node_2, distances[node_1,node_2] )
#                       for node_2 in all_indexes for node_1 in all_indexes])
#     G.remove_edges_from([(idx,idx) for idx in all_indexes])
#     n_pos = {node:np.array([G.nodes[node]['x'],G.nodes[node]['y']],dtype='int') for node in G.nodes}
#     n_col = [G.nodes[node]['color'] for node in G.nodes]
#     nx.draw(G,pos =n_pos,node_color=n_col, with_labels=True)
#     return G




# def find_one_hop_routes(distances,patient_idxs,hospital_start_idxs,hospital_end_idxs,patients):
#     num_options = len(hospital_start_idxs) * len(hospital_end_idxs) * len(patient_idxs)
#     routes = np.zeros([num_options,3],dtype='int')
#     deadlines = np.zeros([num_options],dtype='int')
#     route_distances = np.zeros([num_options],dtype='int')
#     count = 0
#     for triplet in itertools.product(*[hospital_start_idxs,patient_idxs,hospital_end_idxs]):
#         total_dist = distances[triplet[0],triplet[1]] + distances[triplet[1],triplet[2]]
#         if patients[triplet[1],2] > total_dist:
#             routes[count] = np.array(triplet)
#             deadlines[count] = patients[triplet[1],2]
#             route_distances[count] = total_dist
#             count += 1
#     return (routes[:count],deadlines[:count],route_distances[:count])
#
#
# def find_two_hop_routes(distances,patient_idxs,patients,one_hop_routes):
#     existing_routes = one_hop_routes[0]
#     existing_deadlines = one_hop_routes[1]
#     existing_distances = one_hop_routes[2]
#     num_options = existing_routes.shape[0] * len(patient_idxs)
#     routes = np.zeros([num_options,4],dtype='int')
#     deadlines = np.zeros([num_options],dtype='int')
#     route_distances = np.zeros([num_options],dtype='int')
#     count = 0
#     for route_idx, route in enumerate(existing_routes):
#         old_patient = route[1]
#         for new_patient in patient_idxs:
#             if old_patient == new_patient:
#                 continue
#             dist_1 = distances[route[0],route[1]] + distances[route[1],new_patient] + distances[new_patient,route[2]]
#             dist_2 = distances[route[0],new_patient] + distances[new_patient,route[1]] + distances[route[1],route[2]]
#             min_deadline = np.min([existing_deadlines[route_idx], patients[new_patient, 2]])
#             if dist_1 <= dist_2:
#                 if dist_1 <= min_deadline:
#                     routes[count] = [route[0],route[1],new_patient,route[2]]
#                     route_distances[count] = dist_1
#                     deadlines[count] = min_deadline
#                     count += 1
#             else:
#                 if dist_2 <= min_deadline:
#                     routes[count] = [route[0],new_patient,route[1],route[2]]
#                     route_distances[count] = dist_2
#                     deadlines[count] = min_deadline
#                     count += 1
#     return (routes[:count],deadlines[:count],route_distances[:count])
#
#
# def find_three_hop_routes(distances,patient_idxs,patients,two_hop_routes):
#     existing_routes = two_hop_routes[0]
#     existing_deadlines = two_hop_routes[1]
#     existing_distances = two_hop_routes[2]
#     num_options = existing_routes.shape[0] * len(patient_idxs)
#     routes = np.zeros([num_options,5],dtype='int')
#     deadlines = np.zeros([num_options],dtype='int')
#     route_distances = np.zeros([num_options],dtype='int')
#     count = 0
#     for route_idx, route in enumerate(existing_routes):
#         old_patients = set(route[1:3])
#         for new_patient in patient_idxs:
#             if new_patient in old_patients:
#                 continue
#             cut = np.arange(0,3)
#             candidate_dist = np.zeros([len(cut)])
#             candidate_routes= np.zeros([len(cut),5])
#             for cut_idx in cut:
#                 candidate_dist[cut_idx] = existing_distances[route_idx] - distances[route[cut_idx],route[cut_idx+1]] \
#                           + distances[route[cut_idx],new_patient] + distances[new_patient,route[cut_idx+1]]
#                 candidate_routes[cut_idx] = [*route[:cut_idx],new_patient,*route[cut_idx:]]
#
#             min_deadline = np.min([existing_deadlines[route_idx], patients[new_patient, 2]])
#             best_route = candidate_routes[np.argmin(candidate_dist)]
#             if np.min(candidate_dist) <= min_deadline:
#                 routes[count] = best_route
#                 route_distances[count]= np.min(candidate_dist)
#                 deadlines[count] = min_deadline
#                 count += 1
#     return (routes[:count],deadlines[:count],route_distances[:count])
#
#
#     # def all_paths_finder(start_hospital, how_many_patient, distances, patient_idxs, hospital_start_idxs,
#     #                      hospital_end_idxs, patients):
#     #     next_vertices = np.where(distances[start_hospital] > 0)[0]
#     #     num_nodes = patient_idxs.shape[0] ** how_many_patient
#     #     all_paths = np.zeros([distances.shape[0], 3])




# def return_shortest_route(res,distances,hospital_start_idxs,hospital_end_idxs,patients):
#     #still needs shuffling of the patients.
#
#     routes = np.zeros([res.shape[0],res.shape[1]+2],dtype='int')
#     closest_starting_hospitals = hospital_start_idxs[distances[hospital_start_idxs.reshape(-1,1),res[:,0]].argmin(axis=0)]
#     closest_finishing_hospitals = hospital_end_idxs[distances[hospital_end_idxs.reshape(-1,1),res[:,-1]].argmin(axis=0)]
#     routes[:,0] = closest_starting_hospitals
#     routes[:,-1] = closest_finishing_hospitals
#     routes[:,1:-1] = res
#     route_distances = compute_route_distance(routes,distances)
#     route_deadlines = np.min(patients[routes[:,1:-1],2],axis=1)
#     return routes[np.where(route_distances <= route_deadlines)[0]]