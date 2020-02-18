#=========================================================
# NUMERICAL ANALYSIS: USING MESHFREE METHOD

# USING THE REPRODUCING KERNEL COLLOCATION METHOD TO SOLVE 
# THE 2D POISSONS PROBLEM
#=========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import math

# Files
import forming_NS_NC
import forming_A

#-------------------------
#INPUT PARAMETERS
#-------------------------
showPlot = False #Plotting is done if true
printStatements = True #Printing is done if true

#Domain
xdim1 = 0.0
xdim2 = 1.0
ydim1 = 0.0
ydim2 = 1.0

num_pts = np.array([5,5,5,5], float)
#print(num_pts.dtype)

# No. of Source points in the each direction
NS_x = num_pts[0]   # No. of Source points in the x-direction
NS_y = num_pts[1]   # No. of Source points in the y-direction

# No. of Collocation points in the each direction
CP_x  = num_pts[2]  #No. of Collocation points in the x-direction
CP_y  = num_pts[3]  #No. of Collocation points in the y-direction

#-------------------------
# SOURCE POINTS
#-------------------------
NS = forming_NS_NC.forming_SourcePts(xdim1, xdim2, ydim1, ydim2, NS_x, NS_y)


#-------------------------
# COLLOCATION POINTS
#-------------------------
NC, NI_c, NEB = forming_NS_NC.forming_CollocationPts(xdim1, xdim2, ydim1, ydim2, CP_x, CP_y)

#-----------------------------------------------------------------------
# ----------------------------POISSONS----------------------------------
#-----------------------------------------------------------------------

basis = 2   # Code only works for quadratic basis
# No. of source points and total number of collocation points
no_NS = NS.shape[0]  # Getting the number of rows in NS
no_NC = NC.shape[0]
no_NEB = NEB.shape[0]
h = 1/(math.sqrt(no_NS)-1);
ss = (basis+1)*h   # Support size for the RK SF

sq_alphag = no_NS  # Weight for the essential boundary
sq_alphah = 1.0     # Weight for the natural boundary
#-------------------------------------------------------------------------

# Solving the differntial equation
# the A matriz will be of size no_NC x no_NS since u(x,y) is a scalar

#-------------------------
# Forming A matrix
#-------------------------
A1 = forming_A.part_of_NI(NC, NS, ss)







#-------------------------
#PLOTTING
#-------------------------
if showPlot == True:

    plt.figure(1)
    plt.scatter(NS[:,0], NS[:,1], c="r", marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.title("Source Points")
    interactive(True)
    plt.show()

    plt.figure(2)
    plt.scatter(NI_c[:,0], NI_c[:,1], c="b", marker="o", label='Interior')
    plt.scatter(NEB[:,0], NEB[:,1], c="k", marker="o", label='Essential boundary')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.title("Collocation Points")
    plt.legend()
    interactive(False)
    plt.show()

    # For Saving: plt.savefig('ScatterPlot.png')
    # To show all figures at a time:
    # - only first and last figures use interactive, middle ones do not
    # - also first figure should have interactive(True)
    # - and last one should have interactive(False)






