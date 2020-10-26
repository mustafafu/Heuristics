import socket
import time
import sys
import numpy as np



DATA_SIZE=4096
HOST = "localhost"
PORT = int(sys.argv[1])
# PORT = 9000



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            #
            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            # 0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            # 3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_socket(s):
    while True:
        data = s.recv(DATA_SIZE).decode('utf-8')
        if data:
            return data


def send_socket(s, data):
    s.sendall(data.encode('utf-8'))
    return


def parse_walls(current_wall_count,wall_string):
    h_verticals = []
    h_horizontals = []
    h_diags = []
    h_counterdiags = []
    parsed_walls = []
    wall_indexes = []
    wall_idx = 0
    index = 0
    while index < len(wall_string):
        wall_type = wall_string[index]
        index += 1
        if wall_type == 0:
            y1 = wall_string[index]
            index +=1
            y2 = y1
            x1 = wall_string[index]
            index+=1
            x2 = wall_string[index]
            index+=1
            wall = [wall_type,x1,y1,x2,y2]
            h_horizontals.append([x1,y1,x2,y2])
        elif wall_type == 1:
            x1 = wall_string[index]
            index +=1
            x2 = x1
            y1 = wall_string[index]
            index+=1
            y2 = wall_string[index]
            index+=1
            wall = [wall_type,x1,y1,x2,y2]
            h_verticals.append([x1,y1,x2,y2])
        elif wall_type == 2:
            x1 = wall_string[index]
            index += 1
            x2 = wall_string[index]
            index += 1
            y1 = wall_string[index]
            index += 1
            y2 = wall_string[index]
            index += 1
            build = wall_string[index]
            index += 1
            wall = [wall_type,x1,y1,x2,y2]
            h_diags.append([x1,y1,x2,y2])
        elif wall_type == 3:
            x1 = wall_string[index]
            index += 1
            x2 = wall_string[index]
            index += 1
            y1 = wall_string[index]
            index += 1
            y2 = wall_string[index]
            index += 1
            build = wall_string[index]
            index += 1
            wall = [wall_type,x1,y1,x2,y2]
            h_counterdiags.append([x1,y1,x2,y2])
        else:
            print('Unknown wall type')
        parsed_walls.append(wall)
        wall_indexes.append(wall_idx)
        wall_idx += 1
    return parsed_walls,h_horizontals,h_verticals,h_diags,h_counterdiags,wall_indexes


def should_hunter_vertical_build(h_x,h_y,p_x,p_y,h_vx,h_vy):
    if (p_x-2 <= h_x <= p_x-1) and h_vx > 0:
        return True
    elif (p_x+1 <= h_x <= p_x+2) and h_vx < 0:
        return True
    else:
        return False


def should_hunter_horizontal_build(h_x,h_y,p_x,p_y,h_vx,h_vy):
    if (p_y-2 <= h_y <= p_y-1) and h_vy > 0:
        return True
    elif (p_y+1 <= h_y <= p_y+2) and h_vy < 0:
        return True
    else:
        return False


def should_hunter_diagonal_build(h_x,h_y,p_x,p_y,h_vx,h_vy):
    x_diff = p_x - h_x
    y_diff = p_y - h_y
    if 1 <= x_diff - y_diff <= 4 and h_vx > 0 and h_vy < 0:
        return True
    elif -1 >= x_diff - y_diff >= -4  and h_vx < 0 and h_vy > 0:
        return True
    else:
        return False


def should_hunter_counterdiagonal_build(h_x,h_y,p_x,p_y,h_vx,h_vy):
    x_diff = p_x - h_x
    y_diff = p_y - h_y

    if 1 <= y_diff+x_diff <= 4 and h_vx > 0 and h_vy > 0:
        return True
    elif -1 >= x_diff + y_diff >= -4 and h_vx < 0 and h_vy < 0:
        return True
    else:
        return False


def wall_occupies(w_type,w_x1,w_y1,w_x2,w_y2,p_x,p_y):
    if w_type == 0:
        return p_y == w_y1 and min(w_x1, w_x2) <= p_x <= max(w_x2, w_x1)
    elif w_type == 1:
        return p_x == w_x1 and min(w_y1, w_y2) <= p_y <= max(w_y2, w_y1)
    elif w_type==2:
        small_x = min(w_x1,w_x2)
        large_x = max(w_x1,w_x2)
        small_y = min(w_y1,w_y2)
        large_y = max(w_y1,w_y2)
        x_diff = p_x - small_x
        y_diff = p_y - small_y
        # print('x_diff : {}, y_diff : {}'.format(x_diff,y_diff))
        return x_diff <= y_diff <= x_diff + 1 and small_x <= p_x <= large_x and small_y <= p_y <= large_y
    elif w_type == 3:
        small_x = min(w_x1,w_x2)
        large_x = max(w_x1,w_x2)
        small_y = min(w_y1,w_y2)
        large_y = max(w_y1,w_y2)
        x_diff = p_x - small_x
        y_diff = large_y - p_y
        # print('x_diff : {}, y_diff : {}'.format(x_diff, y_diff))
        return x_diff <= y_diff <= x_diff+1 and small_x <= p_x <= large_x and small_y <= p_y <= large_y
    else:
        print('Unknown Wall type in function wall_occupies')
        return


def find_useful_walls(walls,idx_walls,px,py):
    useful_walls = np.zeros(len(idx_walls),dtype=bool)
    [t_lim,b_lim,r_lim,l_lim] = [py,py,px,px]
    [tr_lim,tl_lim,br_lim,bl_lim] = [[px,py],[px,py],[px,py],[px,py]]
    [t_found,b_found,r_found,l_found] = [False,False,False,False]
    [tr_found,tl_found,br_found,bl_found] = [False,False,False,False]

    while not tr_found and tr_lim[0] <= 299 and tr_lim[1] <= 299:
        # print('TR LIM {}'.format(tr_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,tr_lim[0],tr_lim[1]):
                tr_found = True
                useful_walls[i] = True
                # print(wall)
                break
        tr_lim = [_+1 for _ in tr_lim]

    while not tl_found and tl_lim[0] >= 0 and tl_lim[1] <= 299:
        # print('TL LIM {}'.format(tl_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,tl_lim[0],tl_lim[1]):
                tl_found = True
                useful_walls[i] = True
                # print(wall)
                break
        tl_lim = [tl_lim[0] - 1, tl_lim[1] + 1]

    while not bl_found and bl_lim[0] >= 0 and bl_lim[1] >= 0:
        # print('BL LIM {}'.format(bl_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,bl_lim[0],bl_lim[1]):
                bl_found = True
                useful_walls[i] = True
                # print(wall)
                break
        bl_lim = [bl_lim[0] - 1, bl_lim[1] - 1]

    while not br_found and br_lim[0] <= 299 and br_lim[1] >= 0:
        # print('BR LIM {}'.format(br_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,br_lim[0],br_lim[1]):
                br_found = True
                useful_walls[i] = True
                # print(wall)
                break
        br_lim = [br_lim[0] + 1, br_lim[1] - 1]


    while not t_found and t_lim <= 299:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,px,t_lim):
                t_found = True
                useful_walls[i] = True
                # print(wall)
                break
        t_lim += 1
    while not b_found and b_lim >= 0:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,px,b_lim):
                b_found = True
                useful_walls[i] = True
                # print(wall)
                break
        b_lim -= 1
    while not r_found and r_lim <= 299:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,r_lim,py):
                r_found = True
                useful_walls[i] = True
                # print(wall)
                break
        r_lim += 1
    while not l_found and l_lim >= 0:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,l_lim,py):
                l_found = True
                useful_walls[i] = True
                # print(wall)
                break
        l_lim -= 1
    return [walls[i] for i in np.where(useful_walls)[0]], useful_walls


def find_bounding_walls(walls,idx_walls,px,py):
    useful_walls = np.zeros(len(idx_walls),dtype=bool)
    [t_lim,b_lim,r_lim,l_lim] = [py,py,px,px]
    [tr_lim,tl_lim,br_lim,bl_lim] = [[px,py],[px,py],[px,py],[px,py]]
    [t_found,b_found,r_found,l_found] = [False,False,False,False]
    [tr_found,tl_found,br_found,bl_found] = [False,False,False,False]
    [t_d,b_d,r_d,l_d,tr_d,tl_d,br_d,bl_d] = [0,0,0,0,0,0,0,0]

    while not tr_found and tr_lim[0] <= 299 and tr_lim[1] <= 299:
        # print('TR LIM {}'.format(tr_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,tr_lim[0],tr_lim[1]):
                tr_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not tr_found:
            tr_lim = [_+1 for _ in tr_lim]
            tr_d += 1
        else:
            break


    while not tl_found and tl_lim[0] >= 0 and tl_lim[1] <= 299:
        # print('TL LIM {}'.format(tl_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,tl_lim[0],tl_lim[1]):
                tl_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not tl_found:
            tl_lim = [tl_lim[0] - 1, tl_lim[1] + 1]
            tl_d += 1
        else:
            break


    while not bl_found and bl_lim[0] >= 0 and bl_lim[1] >= 0:
        # print('BL LIM {}'.format(bl_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,bl_lim[0],bl_lim[1]):
                bl_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not bl_found:
            bl_lim = [bl_lim[0] - 1, bl_lim[1] - 1]
            bl_d += 1
        else:
            break

    while not br_found and br_lim[0] <= 299 and br_lim[1] >= 0:
        # print('BR LIM {}'.format(br_lim))
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,br_lim[0],br_lim[1]):
                br_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not br_found:
            br_lim = [br_lim[0] + 1, br_lim[1] - 1]
            br_d +=1
        else:
            break


    while not t_found and t_lim <= 299:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,px,t_lim):
                t_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not t_found:
            t_lim += 1
            t_d += 1
        else:
            break

    while not b_found and b_lim >= 0:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,px,b_lim):
                b_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not b_found:
            b_lim -= 1
            b_d += 1
        else:
            break

    while not r_found and r_lim <= 299:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,r_lim,py):
                r_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not r_found:
            r_lim += 1
            r_d += 1
        else:
            break

    while not l_found and l_lim >= 0:
        for i,wall in enumerate(walls):
            if wall_occupies(*wall,l_lim,py):
                l_found = True
                useful_walls[i] = True
                # print(wall)
                break
        if not l_found:
            l_lim -= 1
            l_d += 1
        else:
            break

    return [t_d,b_d,r_d,l_d,tr_d,tl_d,br_d,bl_d],[walls[i] for i in np.where(useful_walls)[0]], useful_walls


def find_hunter_trajectory(hx,hy,hvx,hvy):
    if hvx > 0 and hvy > 0:
        return [[hx+i,hy+i] for i in range(0,300,1) if (hx + i <=299 and hy + i <= 299)]
    elif hvx > 0 and hvy < 0:
        return [[hx + i, hy - i] for i in range(0, 300, 1) if (hx + i <= 299 and hy + i >= 0)]
    elif hvx < 0  and hvy < 0:
        return [[hx - i, hy - i] for i in range(0, 300, 1) if (hx + i >= 0 and hy + i >= 0)]
    elif hvx < 0 and hvy > 0:
        return [[hx - i, hy + i] for i in range(0, 300, 1) if (hx + i >= 0 and hy + i <= 299)]
    else:
        print('Something wrong in find hunter trahectory about speeds')
        return


def avoid_hunter(hx,hy,px,py,hvx,hvy): #hunter_x,hunter_y,prey_x,prey_y,hunter_vx,hunter_vy
    x_diff = px - hx
    y_diff = py - hy
    if hvx > 0 and hvy > 0:
        if x_diff - y_diff > 0:
            return +1,-1
        else:
            return -1,+1
    elif hvx > 0 and hvy < 0:
        if x_diff + y_diff > 0:
            return +1,+1
        else:
            return -1,-1
    elif hvx < 0 and hvy > 0:
        if x_diff + y_diff > 0:
            return +1, +1
        else:
            return -1, -1
    elif hvx < 0  and hvy < 0:
        if x_diff - y_diff > 0:
            return +1,-1
        else:
            return -1,+1
    else:
        print('Issue about speeds in avoid hunter')
        return


def p_dist(x1,y1,x2,y2):
    return np.sqrt(np.square(x2-x1) + np.square(y2 - y1) )


def find_prey_move(walls,idx_walls,px,py,hx,hy,hvx,hvy):
    if p_dist(hx,hy,px,py) < 20:
        x, y = avoid_hunter(hx, hy, px, py, hvx, hvy)
    else:
        [t_d, b_d, r_d, l_d, tr_d, tl_d, br_d, bl_d], bounding_walls, useful_wall_indexes = find_bounding_walls(walls,idx_walls,px,py)
        if t_d >= b_d:
            y = 1
        else:
            y = -1
        if r_d >= l_d:
            x = 1
        else:
            x = -1
    return x,y


def check_if_hunter_stuck(walls,hx,hy,hvx,hvy):
    walls.append([3,299,299,298,300])
    walls.append([3, 0, 0, -1, 1])
    walls.append([2, 0, 299, 1, 300])
    walls.append([2, 299, 0, 300, 1])
    if (hvx > 0 and hvy > 0) or (hvx < 0 and hvy < 0):
        cx, cy = hx, hy
        w1 = -1
        w2 = -1
        w1_idx = -1
        w2_idx = -1
        d_collision = False
        while not d_collision and 0 <= cx <= 299 and 0<= cy <= 299:
            cx, cy = cx + hvx, cy + hvy
            # print('{} {}'.format(cx,cy))
            for i, wall in enumerate(walls):
                if wall_occupies(*wall, cx, cy):
                    d_collision = True
                    w1 = wall[0]
                    w1_idx = i
                    break
        cx,cy = hx,hy
        hvx = -1 * hvx
        hvy = -1 * hvy
        d_collision = False
        while not d_collision and 0 <= cx <= 299 and 0<= cy <= 299:
            cx, cy = cx + hvx, cy + hvy
            # print('{} {}'.format(cx,cy))
            for i, wall in enumerate(walls):
                if wall_occupies(*wall, cx, cy):
                    d_collision = True
                    w2 = wall[0]
                    w2_idx = i
                    break
        if w1 == w2 and w1 == 3:
            return True, min(w1_idx,w2_idx),max(w1_idx,w2_idx)
        else:
            return False, min(w1_idx,w2_idx),max(w1_idx,w2_idx)
    elif (hvx > 0 and hvy < 0) or (hvx < 0 and hvy > 0):
        cx, cy = hx, hy
        w1 = -1
        w2 = -1
        w1_idx = -1
        w2_idx = -1
        cd_collision = False
        while not cd_collision and 0 <= cx <= 299 and 0<= cy <= 299:
            cx, cy = cx + hvx, cy + hvy
            for i, wall in enumerate(walls):
                if wall_occupies(*wall, cx, cy):
                    cd_collision = True
                    w1 = wall[0]
                    w1_idx = i
                    break
        cx,cy = hx,hy
        hvx = -1 * hvx
        hvy = -1 * hvy
        cd_collision = False
        while not cd_collision and 0 <= cx <= 299 and 0<= cy <= 299:
            cx, cy = cx + hvx, cy + hvy
            for i, wall in enumerate(walls):
                if wall_occupies(*wall, cx, cy):
                    cd_collision = True
                    w2 = wall[0]
                    w2_idx = i
                    break
        if w1 == w2 and w1 == 2:
            return True, min(w1_idx,w2_idx),max(w1_idx,w2_idx)
        else:
            return False, min(w1_idx,w2_idx),max(w1_idx,w2_idx)


def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy


def check_if_wall_between(parsed_walls,hx,hy,px,py,hvx,hvy):
    points = [i for i in bresenham(hx,hy,px,py)]
    for point in points:
        ix,iy = point
        for i, wall in enumerate(parsed_walls):
            if wall_occupies(*wall, ix, iy):
                return True, i
    return False,-1


def find_wall_to_left(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cx = cx - 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_right(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cx = cx + 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_top(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cy = cy + 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_bottom(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cy = cy - 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_top_right(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cy = cy + 1
        cx = cx + 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_bottom_right(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cy = cy - 1
        cx = cx + 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_top_left(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cy = cy + 1
        cx = cx - 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx


def find_wall_to_bottom_left(walls,hx,hy):
    cx,cy = hx,hy
    w1_idx = -1
    d_collision = False
    while not d_collision and 0 <= cx <= 299 and 0 <= cy <= 299:
        cy = cy - 1
        cx = cx - 1
        # print('{} {}'.format(cx,cy))
        for i, wall in enumerate(walls):
            if wall_occupies(*wall, cx, cy):
                d_collision = True
                w1_idx = i
                break
    return w1_idx



hunter = False
stream = ""
isFirstRound = True
prev_vx, prev_vy = 0,0
just_removed = -1
while True:
    line = get_socket(s).split("\n")[0]

    val = .01
    time.sleep(val)

    tosend = None

    if line == "done":
        break
    elif line == "hunter":
        hunter = True
    elif line == "prey":
        hunter = False
    elif line == "sendname":
        tosend = "meow_" + str(PORT)
    else:
        data = line.split(" ")

        [rem_time,n_game,n_tick,max_wall,wall_delay,b_X,b_Y,wall_timer,
         hunter_x,hunter_y,hv_x,hv_y,prey_x,prey_y, current_wall_count] = [int(_) for _ in data[:15]]
        # print('Prey_pos: [{},{}], Wall count : {}'.format(prey_x,prey_y,current_wall_count))
        critical_verticals = [0,299]
        critical_horizontals = [0,299]
        critical_diags = []
        critical_counterdiags = []


        wall_string = [int(_) for _ in data[15:]]
        parsed_walls,h_horizontals,h_verticals,h_diags,h_counterdiags,wall_indexes = parse_walls(current_wall_count, wall_string)

        # if len(parsed_walls) > 5:
        #     break

        if hunter:
            wall = "0"
            if just_removed == -1:
                shouldVertical = should_hunter_vertical_build(hunter_x,hunter_y,prey_x,prey_y,hv_x,hv_y)
                shouldHorizontal = should_hunter_horizontal_build(hunter_x, hunter_y, prey_x, prey_y, hv_x, hv_y)
                shouldDiagonal = should_hunter_diagonal_build(hunter_x, hunter_y, prey_x, prey_y, hv_x, hv_y)
                shouldCounterDiagonal = should_hunter_counterdiagonal_build(hunter_x, hunter_y, prey_x, prey_y, hv_x, hv_y)
                if shouldVertical + shouldHorizontal + shouldDiagonal + shouldCounterDiagonal > 1:
                    print('More than one wall is necessary')
                if shouldHorizontal:
                    wall = "1"
                if shouldVertical:
                    wall = "2"
                if shouldDiagonal:
                    wall = "3"
                if shouldCounterDiagonal:
                    wall = "4"
            elif just_removed == 0:
                shouldHorizontal = should_hunter_horizontal_build(hunter_x, hunter_y, prey_x, prey_y, hv_x, hv_y)
                if shouldHorizontal:
                    wall = "1"
                    just_removed = -1
            elif just_removed == 1:
                shouldVertical = should_hunter_vertical_build(hunter_x,hunter_y,prey_x,prey_y,hv_x,hv_y)
                if shouldVertical:
                    wall = "2"
                    just_removed = -1
            elif just_removed == 2:
                shouldDiagonal = should_hunter_diagonal_build(hunter_x, hunter_y, prey_x, prey_y, hv_x, hv_y)
                if shouldDiagonal:
                    wall = "3"
                    just_removed = -1
            elif just_removed == 3:
                shouldCounterDiagonal = should_hunter_counterdiagonal_build(hunter_x, hunter_y, prey_x, prey_y, hv_x, hv_y)
                if shouldCounterDiagonal:
                    wall = "4"
                    just_removed = -1

            # if wall == "0":
            #     willBuild = 0
            # else:
            #     willBuild = 1


            if current_wall_count  >= max_wall:
                if hv_x > 0 and prev_vx < 0 and prev_vy == hv_y and prey_x > hunter_x + 2*wall_timer + 2:
                    wall_to_left = find_wall_to_left(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_left)
                    wall = wall + " " + wtr
                    just_removed = 1
                elif hv_x < 0 and prev_vx > 0 and prev_vy == hv_y and prey_x < hunter_x - 2*wall_timer -2:
                    wall_to_right = find_wall_to_right(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_right)
                    wall = wall + " " + wtr
                    just_removed = 1
                elif hv_y > 0 and prev_vy < 0 and prev_vx == hv_x and prey_y > hunter_y + 2*wall_timer +2:
                    wall_to_bottom = find_wall_to_bottom(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_bottom)
                    wall = wall + " " + wtr
                    just_removed = 0
                elif hv_y < 0 and prev_vy > 0 and prev_vx == hv_x and prey_y < hunter_y - 2*wall_timer -2:
                    wall_to_top = find_wall_to_top(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_top)
                    wall = wall + " " + wtr
                    just_removed = 0
                elif hv_y < 0 and hv_x < 0 and prev_vx>0 and prev_vy>0 and prey_y < hunter_y - 2*wall_timer -2 and prey_x < hunter_x - 2*wall_timer -2:
                    wall_to_top_right = find_wall_to_top_right(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_top_right)
                    wall = wall + " " + wtr
                    # just_removed = 3
                elif hv_y > 0 and hv_x > 0 and prev_vx<0 and prev_vy<0 and prey_y > hunter_y + 2*wall_timer +2 and prey_x > hunter_x + 2*wall_timer +2:
                    wall_to_bottom_left = find_wall_to_bottom_left(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_bottom_left)
                    wall = wall + " " + wtr
                    # just_removed = 3
                elif hv_y > 0 and hv_x < 0 and prev_vx>0 and prev_vy<0 and prey_y > hunter_y + 2*wall_timer +2 and prey_x < hunter_x - 2*wall_timer -2:
                    wall_to_bottom_right = find_wall_to_bottom_right(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_bottom_right)
                    wall = wall + " " + wtr
                    # just_removed = 2
                elif hv_y < 0 and hv_x > 0 and prev_vx < 0 and prev_vy > 0 and prey_y < hunter_y - 2*wall_timer -2 and prey_x > hunter_x + 2*wall_timer +2:
                    wall_to_top_left = find_wall_to_top_left(parsed_walls, hunter_x, hunter_y)
                    wtr = str(wall_to_top_left)
                    wall = wall + " " + wtr
                    # just_removed = 2
                else:
                    useful_walls, useful_wall_indexes = find_useful_walls(parsed_walls, wall_indexes, prey_x, prey_y)
                    remove_wall_idx = [str(i) for i in np.where(~useful_wall_indexes)[0]]
                    for wtr in remove_wall_idx:
                        wall = wall + " " + wtr



            isStuck,w1tr,w2tr = check_if_hunter_stuck(parsed_walls, hunter_x, hunter_y, hv_x, hv_y)
            if isStuck:
                wall = wall + " " + str(w1tr)

            isSeparated,wall_idx_to_rm = check_if_wall_between(parsed_walls, hunter_x, hunter_y, prey_x,prey_y,hv_x,hv_y)
            if isSeparated:
                wall = wall + " " + str(wall_idx_to_rm)

            prev_vx, prev_vy = hv_x, hv_y

            tosend = data[1] + " " + data[2] + " " + wall
        else:
            x,y = find_prey_move(parsed_walls,wall_indexes,prey_x,prey_y,hunter_x,hunter_y,hv_x,hv_y)
            tosend = data[1] + " " + data[2] + " " + str(x) + " " + str(y)

    if tosend is not None:
        # print("sending: {}".format(tosend))
        send_socket(s,tosend+'\n')









