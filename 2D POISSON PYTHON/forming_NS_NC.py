# Funtions for creating source and collocation points

import numpy as np
import math

def forming_SourcePts(xdim1, xdim2, ydim1, ydim2, NS_x, NS_y):
    # Constructing the nodes in the x and y directions
    x_s = np.linspace(xdim1,xdim2,NS_x)
    y_s = np.linspace(ydim1,ydim2,NS_y)

    # Forming all the x,y coordinates of the Source points in the domain
    NP_s = len(x_s)*len(y_s) # Total number of points
    NS = np.zeros((NP_s, 2))
    int_1 = 0
    for count_1 in range(len(x_s)):
        for count_2 in range(len(y_s)):
            NS[int_1,0] += x_s[count_1]
            NS[int_1,1] += y_s[count_2]
            int_1 += 1

    return NS

def forming_CollocationPts(xdim1, xdim2, ydim1, ydim2, CP_x, CP_y):
    # Constructing the nodes in the x and y directions
    x_c = np.linspace(xdim1,xdim2,CP_x)
    y_c = np.linspace(ydim1,ydim2,CP_y)

    # Forming all the x,y coordinates of the Collocation points in the domain
    NP_c = len(x_c)*len(y_c) # Total no. of Collocation points no_NI
    NC_total = np.zeros((NP_c, 2))
    int_1 = 0
    for count_1 in range(len(x_c)):
        for count_2 in range(len(y_c)):
            NC_total[int_1,0] += x_c[count_1]
            NC_total[int_1,1] += y_c[count_2]
            int_1 += 1

    # Splittng the collocation points into interior and boundary points
    # Constructing the INTERIOR COLLOCATION POINTS NI_c

    NP_in = (len(x_c)-2)*(len(y_c)-2)
    NI_c = np.zeros((NP_in, 2))

    # Constructing Nodes on the Essential Boundary
    NP_EB = (2*len(x_c)+2*len(y_c)-4)
    NEB = np.zeros((NP_EB, 2))

    int_2 = 0 
    int_3 = 0
    for int_1 in range(NP_c):
        x_temp = NC_total[int_1,0]
        y_temp = NC_total[int_1,1]
        if (math.isclose(x_temp,xdim1) or math.isclose(x_temp,xdim2) or math.isclose(y_temp,ydim1) or math.isclose(y_temp,ydim2)):
            NEB[int_3,0] += x_temp
            NEB[int_3,1] += y_temp
            int_3 += 1
        else:
            NI_c[int_2,0] += x_temp
            NI_c[int_2,1] += y_temp
            int_2 += 1

    # FORMING THE COLLOCATION POINTS MATRIX 
    NC = np.concatenate((NI_c,NEB))

    return NC, NI_c, NEB
        
            
    

              
            
            
