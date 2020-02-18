# Funtions for creating the A matrix

import numpy as np
import math
import ShapeFunction

# For interior points

import ShapeFunction

def part_of_NI(NC, NS, ss):

    A = np.zeros((NC.shape[0], NS.shape[0]))
    int_row = 0  # Row counter for A
    int_col = 0  # Column counter for A

    #Looping over the no. of collocation pts. in the domain
    for int_1 in range(NC.shape[0]):
        # arranging rows of A
        # We need the RK shape function at the collocation point x,y centered at the Source point
        x = NC[int_1,0]
        y = NC[int_1,1]
        P = ShapeFunction.required_nodes(x,y,NS,ss)
        SIxx = ShapeFunction.DSF_xx(x,y,NS,P,ss)
        SIyy = ShapeFunction.DSF_yy(x,y,NS,P,ss)





    return A

