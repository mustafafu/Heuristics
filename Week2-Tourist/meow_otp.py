from site_object import Site, two_opt_algorithm
import numpy as np
import os
import seaborn as sns
from sklearn.cluster import KMeans
import hungarian_algorithm as hng
import sys
import time

# This functions shows how to read the data from stdin using the specified
# format, but it does not save any useful information other than the number
# of sites and the number of days. The data should be saved are marked
# between vvvvvv and ^^^^^^. Taken from the architect team git,
# https://github.com/EAirPeter/HPS-OptimalTouring-Arch/blob/master/solvers/sample-python/solver.py
def read_all():
    state = 0
    n_site = 0
    n_day = 0
    data_entry_values = []
    data_entry_hours = []



    # for line in sys.stdin:

    # Using readlines()
    file1 = open('touringtest2.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        ln = line.split()
        if not ln:
            continue
        if state == 0:
            # Ignore the line with column names of the first part
            # site avenue street desiredtime value
            state = 1
        elif state == 1:
            if ln[0][0].isdigit():
                # vvvvvvv save these
                site = int(ln[0])
                avenue = int(ln[1])
                street = int(ln[2])
                desired_time = int(ln[3])
                value = float(ln[4])
                data_entry_values.append([site,avenue,street,desired_time,value])
                # ^^^^^^
                n_site = max(n_site, site)
            else:
                # Ignore the line with column names of the second part
                # site day beginhour endhour
                state = 2
        elif state == 2:
            # vvvvvv You may want to save these data
            site = int(ln[0])
            day = int(ln[1])
            begin_hour = int(ln[2])
            end_hour = int(ln[3])
            data_entry_hours.append([site,day,begin_hour,end_hour])
            # ^^^^^^
            n_day = max(n_day, day)

    site_values = np.zeros([n_site])
    site_required_time = np.zeros([n_site])
    site_locations = np.zeros([n_site,2])
    site_beginhours = np.zeros([n_day,n_site])
    site_endhours = np.zeros([n_day,n_site])
    site_names = np.zeros([n_site])

    for entry in data_entry_values:
        site_idx = int(entry[0] - 1)
        site_names[site_idx] = entry[0]
        site_values[site_idx] = entry[4]
        site_required_time[site_idx] = entry[3]
        site_locations[site_idx,:] = [entry[1],entry[2]]

    for entry in data_entry_hours:
        day_idx = int(entry[1] - 1)
        site_idx = int(entry[0] - 1)
        site_beginhours[day_idx,site_idx] = entry[2]
        site_endhours[day_idx,site_idx] = entry[3]

    return n_site, n_day, site_names ,site_values, site_required_time , site_locations, site_beginhours, site_endhours


def compute_value_of_day(day_idx, list_of_sites, site_beginhours, site_endhours, site_required_time, site_values, site_locations):
    current_val = 0
    current_x = 0
    current_y = 0
    current_time = 0
    visited_sites = set()
    is_first_site_of_day = True
    feasible_list = []
    for site in list_of_sites:
        site_idx = int(site) - 1
        site_x, site_y = site_locations[site_idx, :]
        if site_idx in visited_sites:
            raise RuntimeError('You have already visited site {}'.format(site_idx))
        else:
            visited_sites.add(site_idx)
        if is_first_site_of_day:
            current_x = site_x
            current_y = site_y
            current_time = site_beginhours[day_idx][site_idx] * 60
            is_first_site_of_day = False
            if current_time + site_required_time[site_idx] > site_endhours[day_idx][site_idx] * 60:
                # print('Insufficient time to visit site {}'.format(site_idx))
                continue
            else:
                feasible_list.append(int(site))
                current_val += site_values[site_idx]
                current_time += site_required_time[site_idx]
        else:
            travel_time = abs(current_x - site_x) + abs(current_y - site_y)
            visit_time = site_required_time[site_idx]
            possible_leaving_time = max(current_time + travel_time, site_beginhours[day_idx][site_idx] * 60) + visit_time
            # print(possible_leaving_time)
            if possible_leaving_time > site_endhours[day_idx][site_idx] * 60:
                # print('Insufficient time to visit site {}'.format(site_idx))
                continue
            elif (current_time + travel_time) < (site_beginhours[day_idx][site_idx] * 60):
                continue
            else:
                feasible_list.append(int(site))
                # print('Site {}, BeginHours {}, Endhour {}, Arrive {}, Leave {}'.format(site,site_beginhours[day_idx][site_idx],site_endhours[day_idx][site_idx],max(current_time + travel_time, site_beginhours[day_idx][site_idx]*60),max(current_time + travel_time, site_beginhours[day_idx][site_idx])+ visit_time ))
                current_x = site_x
                current_y = site_y
                current_val += site_values[site_idx]
                current_time = possible_leaving_time
    return current_val, feasible_list



def tsp_and_brute(maximal_matching,grouped_sites,site_beginhours,site_endhours, site_required_time, site_values, site_locations):
    visiting_order = []
    for i in range(len(grouped_sites)):
        num_sites = len(grouped_sites[i])
        visiting_order.append(two_opt_algorithm(grouped_sites[i], num_sites))

    overall_solution = [i for i in range(len(maximal_matching))]
    overall_values = [i for i in range(len(maximal_matching))]
    day_list = []
    day_value = []
    for match in maximal_matching:
        group_idx = int(match[0][0].split(sep='_')[1])
        day_idx = int(match[0][1].split(sep='_')[1])
        todays_sites = np.array([site.id for site in visiting_order[group_idx]])
        best_list_so_far = []
        best_value_so_far = 0
        for start_idx in range(len(todays_sites)):
            rolled_list = np.roll(todays_sites, -1 * start_idx)
            current_val, feasible_list = compute_value_of_day(day_idx, rolled_list, site_beginhours, site_endhours,
                                                              site_required_time, site_values, site_locations)
            if current_val > best_value_so_far:
                best_value_so_far = current_val
                best_list_so_far = feasible_list
        flipped_today_sites = np.flip(todays_sites)
        for start_idx in range(len(flipped_today_sites)):
            rolled_list = np.roll(flipped_today_sites, -1 * start_idx)
            current_val, feasible_list = compute_value_of_day(day_idx, rolled_list, site_beginhours, site_endhours,
                                                              site_required_time, site_values, site_locations)
            if current_val > best_value_so_far:
                best_value_so_far = current_val
                best_list_so_far = feasible_list

        # print('On day {} visit sites :{}'.format(day_idx+1,best_list_so_far))
        # print(*best_list_so_far)
        # print('Value : {}'.format(best_value_so_far))
        overall_values[day_idx] = best_value_so_far
        overall_solution[day_idx] = np.array(best_list_so_far)
    return overall_solution,overall_values


def main():
    return True

if __name__ == '__main__':

    n_site, n_day, site_names, site_values, site_required_time, site_locations, site_beginhours, site_endhours = read_all()

    time_0 = time.time()

    best_value_so_far = []
    best_tour_so_far = []
    split_iter = 0

    while time.time() - time_0 < 90:
        split_iter += 1
        kmeans = KMeans(n_clusters=n_day).fit(site_locations)

        total_time = np.zeros([n_day, n_day])

        grouped_sites = []
        for group in np.unique(kmeans.labels_):
            this_group_sites = []
            group_sites_idx = np.where(kmeans.labels_ == group)[0]
            for site_idx in group_sites_idx:
                site_id = site_names[site_idx]
                this_group_sites.append(Site(site_id, site_locations[site_idx, 0], site_locations[site_idx, 1]))
            grouped_sites.append(this_group_sites)
            for dd in range(n_day):
                total_time[group, dd] = np.sum(site_endhours[dd, group_sites_idx] - site_beginhours[dd, group_sites_idx])

        total_time_dictionary = {'group_' + str(ii): {'day_' + str(jj): value for jj, value in enumerate(group_array)}
                                 for ii, group_array in enumerate(total_time)}
        maximal_matching = hng.find_matching(total_time_dictionary, matching_type='max', return_type='list')

        # print(maximal_matching)

        overall_solution_tsp,overall_values_tsp= tsp_and_brute(maximal_matching,grouped_sites,site_beginhours,site_endhours,
                                                               site_required_time, site_values, site_locations)

        if np.sum(overall_values_tsp) > np.sum(best_value_so_far):
            best_value_so_far = overall_values_tsp
            best_tour_so_far = overall_solution_tsp
            # print('For iteration {} at time {}'.format(split_iter,time.time()-time_0))
            # print(maximal_matching)
            for ii in range(n_day):
                current_val, feasible_list = compute_value_of_day(ii, list(overall_solution_tsp[ii]), site_beginhours,
                                                                  site_endhours,
                                                                  site_required_time, site_values, site_locations)
                # print('Day {} :::: TSP Solution Value: {}, Visiting Order: {}'.format(ii + 1, current_val, feasible_list))
        # just to make sure if first iteration is too long for some reason don't continue iterating
        # print one solution at least
        if split_iter == 1:
            if time.time()-time_0 > 55:
                for day in range(n_day):
                    print(*list(best_tour_so_far[day]))

    for day in range(n_day):
        print(*list(best_tour_so_far[day]))