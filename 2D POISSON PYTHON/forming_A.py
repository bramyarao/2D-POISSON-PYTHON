# Funtions for creating the A matrix

import numpy as np
import math
import ShapeFunction

# For interior points

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

        #Looping over the no. of source pts.
        for int_2 in range(NS.shape[0]): 
            a_inter = SIxx[0,int_2] + SIyy[0,int_2]

            # Arranging in in the matrix
            A[int_row,int_col] = A[int_row,int_col] + a_inter
            int_col += 1

        int_col = 0
        int_row += 1

    return A

def part_of_NEB(NEB,NS,ss,sq_alphag):
    A = np.zeros((NEB.shape[0], NS.shape[0]))
    int_row = 0  # Row counter for A
    int_col = 0  # Column counter for A

    #Looping over the no. of boundary collocation pts. in the domain
    for int_1 in range(NEB.shape[0]):
        x = NEB[int_1,0]
        y = NEB[int_1,1]
        P = ShapeFunction.required_nodes(x,y,NS,ss)
        SI = ShapeFunction.SF_2D(x,y,NS,P,ss)

        for int_2 in range(NS.shape[0]):
            a_inter = sq_alphag * SI[0,int_2]   # weight on the EB

            # Arranging in in the matrix
            A[int_row,int_col] = A[int_row,int_col] + a_inter
            int_col += 1

        int_col = 0
        int_row += 1

    return A
