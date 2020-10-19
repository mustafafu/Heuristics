import numpy as np
import argparse
import itertools
import time
import multiprocessing
from multiprocessing import Pool



def read_all(hospital_locations,file_string):
    n_patient = 0
    n_hospital = 0
    hospitals = []
    patients = []
    state = 0
    # for line in sys.stdin:
    file1 = open(file_string, 'r')
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


def compute_cross_distance(array_x_y):
    x_diff = np.abs(array_x_y[:,0].reshape(-1,1) - array_x_y[:,0].reshape(1,-1))
    y_diff = np.abs(array_x_y[:,1].reshape(-1,1) - array_x_y[:,1].reshape(1,-1))
    return x_diff + y_diff


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


def return_shortest_route_w_tsp_fixed_hospital(res,distances,hospital_idx,patients):
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
            routes[:,0] = hospital_idx
            routes[:,-1] = hospital_idx
            routes[:,1:-1] = tmp_res
            route_distances = compute_route_distance(routes,distances)
            if first_time:
                route_deadlines = np.min(patients[routes[:,1:-1],2],axis=1)
                first_time=False
            better_routes_idxs = np.where(min_dists > route_distances)[0]
            min_dists[better_routes_idxs] = route_distances[better_routes_idxs]
            best_routes[better_routes_idxs,:] = routes[better_routes_idxs,:]
    return (best_routes[np.where(min_dists <= route_deadlines)[0]],route_deadlines[np.where(min_dists <= route_deadlines)[0]],min_dists[np.where(min_dists <= route_deadlines)[0]])

    # remaining_route_idxs = remaining_route_idxs[np.argsort(route_deadlines[remaining_route_idxs] - route_distances[remaining_route_idxs])]
    # remaining_route_idxs = remaining_route_idxs[::-1]


def get_hospital_routes(hospital_idx, patient_hospital_match, distances, hospital_start_idxs, patients):
    this_hospital_patients_idx = np.where(patient_hospital_match == hospital_idx)[0]
    n_patients = this_hospital_patients_idx.shape[0]
    route_possibility_count = n_patients * (1 + (n_patients - 1) * (1 + (n_patients - 2) * (1 + (n_patients - 3))))
    routes = -1 * np.ones([route_possibility_count, 6], dtype='int16')
    route_deadlines = -1 * np.ones(route_possibility_count, dtype='int16')
    route_distances = -1 * np.ones(route_possibility_count, dtype='int16')
    routes_idx = 0

    number_of_patients = 1
    aaa = [this_hospital_patients_idx] * number_of_patients
    params = [x for x in itertools.product(*aaa) if len(x) == len(set(x))]
    res = np.array(params, dtype=np.int16)
    res.sort(axis=1)
    res = np.unique(res, axis=0)
    this_routes, this_route_deadlines, this_route_distances = return_shortest_route_w_tsp_fixed_hospital(res, distances,
                                                                                                         hospital_start_idxs[
                                                                                                             hospital_idx],
                                                                                                         patients)
    n_routes = this_routes.shape[0]
    routes[routes_idx:routes_idx + n_routes, :number_of_patients + 2] = this_routes
    route_deadlines[routes_idx:routes_idx + n_routes] = this_route_deadlines
    route_distances[routes_idx:routes_idx + n_routes] = this_route_distances
    routes_idx += n_routes

    number_of_patients += 1
    while number_of_patients <= 4:
        # print(number_of_patients)
        feasible_patients = this_routes[:, 1:-1]
        additional_patients_combos = [(*y[0], y[1]) for y in
                                      itertools.product(feasible_patients.tolist(), this_hospital_patients_idx)]
        additional_patients_combos = [combo for combo in additional_patients_combos if len(combo) == len(set(combo))]
        t0 = time.time()
        res = np.array(additional_patients_combos, dtype=np.int16)
        if len(additional_patients_combos) == 0:
            number_of_patients += 1
            continue
        res.sort(axis=1)
        # print(time.time() - t0)
        t0 = time.time()
        this_routes, this_route_deadlines, this_route_distances = return_shortest_route_w_tsp_fixed_hospital(res,
                                                                                                             distances,
                                                                                                             hospital_start_idxs[
                                                                                                                 hospital_idx],
                                                                                                             patients)
        # print(time.time() - t0)
        n_routes = this_routes.shape[0]
        routes[routes_idx:routes_idx + n_routes, :number_of_patients + 2] = this_routes
        route_deadlines[routes_idx:routes_idx + n_routes] = this_route_deadlines
        route_distances[routes_idx:routes_idx + n_routes] = this_route_distances
        routes_idx += n_routes
        number_of_patients += 1
    routes = routes[:routes_idx]
    route_deadlines = route_deadlines[:routes_idx]
    route_distances = route_distances[:routes_idx]
    how_many_patients = np.ones(route_deadlines.shape)
    for i in range(3, 6):
        how_many_patients[np.where(routes[:, i] > 0)[0]] += 1

    return routes, route_deadlines, route_distances, how_many_patients


def best_value_routes(fnc_routes, fnc_route_distances, fnc_how_many_patients):
    fp_sorted = np.argsort(fnc_route_distances / fnc_how_many_patients)
    scheduled_patients = set()
    selected_routes = []
    route_counter = 0
    for index in fp_sorted:
        candidate_route_w_minus = fnc_routes[index]
        candidate_route = candidate_route_w_minus[np.where(candidate_route_w_minus > -1)[0]]
        if len(scheduled_patients.intersection(candidate_route)) == 0:
            selected_routes.append(index)
            # print(candidate_route)
            route_counter += 1
            for patient in candidate_route[1:-1]:
                scheduled_patients.add(patient)
        else:
            continue
    return np.array(selected_routes)


def greedy_scheduler_backwards(max_time,routes, route_deadlines, route_distances, how_many_patients, scheduled_patients):
    current_deadlines = route_deadlines.copy()
    current_deadlines[np.where(route_deadlines > max_time)[0]] = max_time
    remaining_route_idxs = np.where(route_distances <= max_time)[0]
    if len(remaining_route_idxs) == 0:
        return []
    # remaining_route_idxs = remaining_route_idxs[np.argsort(max_time - (route_deadlines[remaining_route_idxs] - route_distances[remaining_route_idxs]) / how_many_patients[remaining_route_idxs])]
    sorted_indexes = remaining_route_idxs[np.argsort(max_time - (current_deadlines[remaining_route_idxs] - route_distances[remaining_route_idxs]) / how_many_patients[remaining_route_idxs])]
    # sorted_indexes = np.argsort(max_time - (route_deadlines - route_distances) / how_many_patients)
    candidate_route_counter = 0
    candidate_route_w_minus = routes[sorted_indexes[candidate_route_counter]]
    candidate_route = candidate_route_w_minus[np.where(candidate_route_w_minus > -1)[0]]
    while len(scheduled_patients.intersection(candidate_route)) > 0 and candidate_route_counter < len(sorted_indexes) - 1:
        candidate_route_counter +=1
        candidate_route_w_minus = routes[sorted_indexes[candidate_route_counter]]
        candidate_route = candidate_route_w_minus[np.where(candidate_route_w_minus > -1)[0]]
    if len(scheduled_patients.intersection(candidate_route)) == 0:
        next_max_time = current_deadlines[sorted_indexes[candidate_route_counter]] - route_distances[sorted_indexes[candidate_route_counter]]
        next_scheduled_patients = set(candidate_route[1:-1]).union(scheduled_patients)
        # for patient in candidate_route[1:-1]:
        #     scheduled_patients.add(patient)
        other_route_idxs = greedy_scheduler_backwards(next_max_time, routes, current_deadlines, route_distances, how_many_patients, next_scheduled_patients)
        other_route_idxs.append(sorted_indexes[candidate_route_counter])
        return other_route_idxs
    else:
        return []


def greedy_scheduler_forwards(current_time, routes, route_deadlines, route_distances, scheduled_patients):
    remaining_route_idxs = np.where(route_deadlines - route_distances >= current_time)[0]
    if len(remaining_route_idxs) == 0:
        return [], 0
    best_value = 0
    best_other_routes = []
    best_this_route = []
    for ii,candidate_route_w_minus in enumerate(routes[remaining_route_idxs]):
        candidate_route = candidate_route_w_minus[np.where(candidate_route_w_minus > -1)[0]]
        if len(scheduled_patients.intersection(candidate_route)) > 0:
            continue
        next_time = current_time + route_distances[remaining_route_idxs[ii]]
        next_scheduled_patients = set(candidate_route[1:-1]).union(scheduled_patients)
        following_routes, value_forward = greedy_scheduler_forwards(next_time, routes, route_deadlines, route_distances, next_scheduled_patients)
        my_value = value_forward + len(candidate_route[1:-1])
        if my_value > best_value:
            best_value = my_value
            best_other_routes = following_routes
            best_this_route = [remaining_route_idxs[ii]]
            # this_route = other_routes + [candidate_route.tolist()]
    for other_route in best_other_routes:
        best_this_route.append(other_route)
    return best_this_route, best_value


def method_backwards(routes, route_deadlines, route_distances,how_many_patients, scheduled_patients):
    max_time = np.max(route_deadlines)
    selected_routes = greedy_scheduler_backwards(max_time, routes, route_deadlines, route_distances,how_many_patients, scheduled_patients)
    best_value = 0
    for route_idx in selected_routes:
        chosen_route_w_minus = routes[route_idx]
        chosen_route = chosen_route_w_minus[np.where(chosen_route_w_minus > -1)[0]]
        for patient in chosen_route[1:-1]:
            best_value += 1
    return (selected_routes,best_value)


def method_forwards(selected_routes,routes,route_deadlines,route_distances,this_scheduled_patients):
    # #setup for method_forwards
    # current_time = 0
    # this_routes = routes[selected_routes]
    # this_route_deadlines = route_deadlines[selected_routes]
    # this_route_distances = route_distances[selected_routes]
    # best_routes, forward_method_value = greedy_scheduler_forwards(current_time, this_routes, this_route_deadlines,this_route_distances, scheduled_patients)
    # forward_method_routes = selected_routes[best_routes]
    # setup for method_forwards
    current_time = 0
    this_routes = routes[selected_routes]
    this_route_deadlines = route_deadlines[selected_routes]
    this_route_distances = route_distances[selected_routes]
    best_routes, forward_method_value = greedy_scheduler_forwards(current_time, this_routes, this_route_deadlines,this_route_distances, this_scheduled_patients)
    return (selected_routes[best_routes], forward_method_value)


def route_hospital(hospital_idx, patient_hospital_match, distances, hospital_start_idxs, patients, max_ambulance):
    routes, route_deadlines, route_distances, how_many_patients = get_hospital_routes(hospital_idx, patient_hospital_match, distances, hospital_start_idxs, patients)
    selected_routes = best_value_routes(routes, route_distances, how_many_patients)
    scheduled_patients = set()
    each_ambulance_route = []
    this_ambulance = 1
    # while this_ambulance <= hospitals[hospital_idx,2]:
    while this_ambulance <= max_ambulance:
        # p_fwd = Process(target=method_forwards, args=(selected_routes, routes, route_deadlines,route_distances, scheduled_patients))
        # p_fwd.start()
        # p_bwd = Process(target=method_forwards, args=(selected_routes, routes, route_deadlines,route_distances, scheduled_patients))
        # p_bwd.start()
        # p_fwd.join()
        # p_bwd.join()
        (forward_method_routes, forward_method_value) = method_forwards(selected_routes, routes, route_deadlines,route_distances, scheduled_patients)
        (backward_method_routes, backward_method_value) = method_backwards(routes, route_deadlines, route_distances,how_many_patients, scheduled_patients)
        # print('Forward method {}'.format(routes[forward_method_routes]))
        # print('Backward method {}'.format(routes[backward_method_routes]))
        if forward_method_value < backward_method_value:
            this_ambulance_routes = backward_method_routes
            this_ambulance_value = backward_method_value
        else:
            this_ambulance_routes = forward_method_routes
            this_ambulance_value = forward_method_value
        each_ambulance_route.append((this_ambulance_routes, this_ambulance_value))
        # print('---------Ambulance {}---------'.format(this_ambulance))
        for route_idx in this_ambulance_routes:
            chosen_route_w_minus = routes[route_idx]
            chosen_route = chosen_route_w_minus[np.where(chosen_route_w_minus > -1)[0]]
            # print('Route: {}, How long: {}, Deadline:{}'.format(chosen_route, route_distances[route_idx],route_deadlines[route_idx]))
            for patient in chosen_route[1:-1]:
                scheduled_patients.add(patient)
        this_ambulance += 1
    routes_to_cover = []
    for ambulance_route_idxs, ambulance_route_value in each_ambulance_route:
        routes_to_cover.append([routes[ambulance_route_idxs],ambulance_route_value])
    return routes_to_cover


def route_hospital_wrapper(hospital_idx):
    global patient_hospital_match
    global distances
    global hospital_start_idxs
    global patients
    global max_ambulance
    return route_hospital(hospital_idx, patient_hospital_match, distances, hospital_start_idxs, patients, max_ambulance)


def ambulance_match_eval(ambulance_values,ambulance_match):
    total_value = 0
    for hh,ambulance_count in enumerate(ambulance_match):
        total_value += np.sum(ambulance_values[hh,:ambulance_count])
    return total_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data50',
                        help='What is the data file (string)?')
    parser.add_argument('--output', type=str, default='out50',
                        help='What is the output file (string)?')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = main()
    input_file = args.input
    output_file = args.output
    patients, hospitals = read_all(False,input_file)
    patients = np.array(patients)
    hospitals = np.array(hospitals)
    V_c = patients.shape[0]
    W = hospitals.shape[0]
    patient_idxs = np.arange(0, V_c)
    hospital_start_idxs = np.arange(V_c, V_c + W)
    hospital_end_idxs = np.arange(V_c + W, V_c + W + W)
    all_indexes = np.array([*patient_idxs, *hospital_start_idxs, *hospital_end_idxs])
    num_hospitals = hospitals.shape[0]

    if hospitals[0, 0] == -1:
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=num_hospitals).fit(patients[:, :2])
        # hospital_locations = np.round(kmeans.cluster_centers_).astype('int')
        # hospitals[:, :2] = hospital_locations
        from sklearn_extra.cluster import KMedoids
        kmedoids = KMedoids(n_clusters=num_hospitals, metric='manhattan').fit(patients[:, :2])
        hospital_locations = np.round(kmedoids.cluster_centers_).astype('int')
        hospitals[:, :2] = hospital_locations
        # hospitals[:,:2] = np.array([[38,63],
        #                    [65,53],
        #                    [50,42],
        #                    [38,51],
        #                    [48,53]])
        # # 38, 63, 3
        # # 65, 53, 5
        # # 50, 42, 8
        # # 38, 51, 8
        # # 48, 53, 4
    else:
        hospital_locations = hospitals[:, :2]

    all_locations_matrix = np.zeros([V_c + W + W, 2], dtype='int')
    all_locations_matrix[:V_c, :] = patients[:, :2]
    all_locations_matrix[V_c:V_c + W, :] = hospitals[:, :2]
    all_locations_matrix[V_c + W:V_c + W + W, :] = hospitals[:, :2]
    distances = compute_cross_distance(all_locations_matrix)
    distances += 1
    for i in all_indexes:
        distances[i][i] = 0

    patient_hospital_distance = distances[patient_idxs.reshape(-1, 1), hospital_start_idxs.reshape(1, -1)]
    # print('Kmeans distances sum {}'.format(patient_hospital_distance.min(axis=1).sum()))
    patient_hospital_match = np.argmin(patient_hospital_distance, axis=1)

    for hospital_idx in range(num_hospitals):
        this_patients = np.where(patient_hospital_match==hospital_idx)[0]
        # print(np.sum(patient_hospital_match==hospital_idx))
        sorted_distance_args = np.argsort(patient_hospital_distance[this_patients,hospital_idx])
        patients_to_remove = this_patients[sorted_distance_args][75:]
        patient_hospital_match[patients_to_remove] = num_hospitals + 1
        # print(np.sum(patient_hospital_match==hospital_idx))



    max_ambulance = hospitals[:, 2].max()

    ambulance_values = np.zeros([num_hospitals, max_ambulance])

    t0 = time.time()
    num_cpu = multiprocessing.cpu_count()
    p = Pool(num_cpu)
    hospital_routings = p.map(route_hospital_wrapper, (hospital_idx for hospital_idx in range(num_hospitals)))
    # print('Hospitals ready took {}'.format(time.time()-t0))

    for hospital_idx in range(num_hospitals):
        hospital_routing = hospital_routings[hospital_idx]
        for rout_idx, (route, value) in enumerate(hospital_routing):
            # print('Ambulance {} is {} and has value: {}'.format(rout_idx,route,value))
            ambulance_values[hospital_idx, rout_idx] = value

    ambulance_counts = hospitals[:, 2].astype('int16')
    list = [np.arange(0, num_hospitals, dtype='int16')] * num_hospitals
    best_value = 0
    best_match = []
    best_permutation = []
    for permutation in itertools.product(*[*list]):
        if len(permutation) == len(set(permutation)):
            ambulance_match = ambulance_counts[np.array(permutation)]
            this_value = ambulance_match_eval(ambulance_values, ambulance_match)
            if this_value > best_value:
                best_value = this_value
                best_match = ambulance_match
                best_permutation = permutation



    fin = open(output_file, "at")

    # best_permutation = np.arange(num_hospitals)
    tmp_hospitals = hospitals.copy()
    tmp_hospitals = tmp_hospitals[np.argsort(best_permutation)]
    hospitals[:,:2] = tmp_hospitals[:,:2]


    for hospital_idx in range(num_hospitals):
        # print('Hospital:{},{},{}'.format(hospitals[hospital_idx, 0], hospitals[hospital_idx, 1],best_match[hospital_idx]))
        # print('Hospital:{},{},{}'.format(hospitals[hospital_idx, 0], hospitals[hospital_idx, 1],hospitals[hospital_idx, 2]))
        fin.write('Hospital:{},{},{}\n'.format(hospitals[hospital_idx, 0], hospitals[hospital_idx, 1],hospitals[hospital_idx, 2]))

    # print('\n')
    fin.write('\n')
    score = 0
    for h_idx,hospital in enumerate(hospitals):
        number_of_ambulance = hospital[2]
        hospital_routing = hospital_routings[np.argsort(best_permutation)[h_idx]]
        this_hospital_routes = []
        this_hospital_route_start_times = []
        for ambulance in range(number_of_ambulance):
            routes_w_minus = hospital_routing[ambulance][0]
            start_time = 0
            for route_idx,route_w_minus in enumerate(routes_w_minus):
                route = route_w_minus[np.where(route_w_minus > -1)[0]].reshape(1,-1)
                this_hospital_routes.append(route)
                this_hospital_route_start_times.append(start_time)
                start_time += int(compute_route_distance(route,distances))
        for sorted_route_idx in np.argsort(this_hospital_route_start_times):
            this_route = this_hospital_routes[sorted_route_idx][0]
            my_string = 'Ambulance: {}: ({},{})'.format(h_idx + 1, hospital[0],hospital[1])
            for patient in this_route[1:-1]:
                score +=1
                my_string = my_string + ', {}: ({},{},{})'.format(patient + 1, patients[patient, 0],patients[patient, 1], patients[patient, 2])
            my_string = my_string + ', {}: ({},{})'.format(h_idx + 1, hospital[0],hospital[1])
            # print(my_string)
            fin.write(my_string + '\n')

    fin.close()

    print(score)



















