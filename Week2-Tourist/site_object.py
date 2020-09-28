'''
    Description: The Site object for 2-opt traveling salesman
    Author: gumpu - edited by Mustafa accordingly for our scenario
    GitHub: https://github.com/gumpu/TSP_Animation
'''
import numpy as np

class Site( object ):
    def __init__( self, id, x, y ):
        self.id  = id
        self.x   = x
        self.y   = y


def travel_time(s1, s2):
    """Compute the travelling time between two sites."""
    return np.abs(s1.x-s2.x) + np.abs(s1.y-s2.y)




#-----------------------------------------------------------------------------
#
#    Before 2opt             After 2opt
#       Y   Z                    Y   Z
#       O   O----->              O-->O---->
#      / \  ^                     \
#     /   \ |                      \
#    /     \|                       \
# ->O       O              ->O------>O
#   C       X                C       X
#
# In a 2opt optimization step we consider two nodes, Y and X.  (Between Y
# and X there might be many more nodes, but they don't matter.) We also
# consider the node C following Y and the node Z following X. i
#
# For the optimization we see replacing the edges CY and XZ with the edges CX
# and YZ reduces the length of the path  C -> Z.  For this we only need to
# look at |CY|, |XZ|, |CX| and |YZ|.   |YX| is the same in both
# configurations.
#
# If there is a length reduction we swap the edges AND reverse the direction
# of the edges between Y and X.
#
# In the following function we compute the amount of reduction in length
# (gain) for all combinations of nodes (X,Y) and do the swap for the
# combination that gave the best gain.
#

def optimize2opt(nodes, solution, number_of_nodes):
    best = 0
    best_move = None
    # For all combinations of the nodes
    for ci in range(0, number_of_nodes):
        for xi in range(0, number_of_nodes):
            yi = (ci + 1) % number_of_nodes  # C is the node before Y
            zi = (xi + 1) % number_of_nodes  # Z is the node after X

            c = solution[ ci ]
            y = solution[ yi ]
            x = solution[ xi ]
            z = solution[ zi ]
            # Compute the lengths of the four edges.
            cy = travel_time( c, y )
            xz = travel_time( x, z )
            cx = travel_time( c, x )
            yz = travel_time( y, z )

            # Only makes sense if all nodes are distinct
            if xi != ci and xi != yi:
                # What will be the reduction in length.
                gain = (cy + xz) - (cx + yz)
                # Is is any better then best one sofar?
                if gain > best:
                    # Yup, remember the nodes involved
                    best_move = (ci,yi,xi,zi)
                    best = gain

    # print(best_move, best)
    if best_move is not None:
        (ci,yi,xi,zi) = best_move
        # This four are needed for the animation later on.
        c = solution[ ci ]
        y = solution[ yi ]
        x = solution[ xi ]
        z = solution[ zi ]

        # Create an empty solution
        new_solution = [ _ for _ in range(0,number_of_nodes)]
        # In the new solution C is the first node.
        # This we we only need two copy loops instead of three.
        new_solution[0] = solution[ci]

        n = 1
        # Copy all nodes between X and Y including X and Y
        # in reverse direction to the new solution
        while xi != yi:
            new_solution[n] = solution[xi]
            n = n + 1
            xi = (xi-1)%number_of_nodes
        new_solution[n] = solution[yi]

        n = n + 1
        # Copy all the nodes between Z and C in normal direction.
        while zi != ci:
            new_solution[n] = solution[zi]
            n = n + 1
            zi = (zi+1)%number_of_nodes
        return (True,new_solution)
    else:
        return (False,solution)



#-----------------------------------------------------------------------------
def two_opt_algorithm(sites, number_of_nodes):
    # Create an initial solution
    solution = [n for n in sites]
    go = True
    # Try to optimize the solution with 2opt until
    # no further optimization is possible.
    while go:
        (go,solution) = optimize2opt(sites, solution, number_of_nodes)
    return solution