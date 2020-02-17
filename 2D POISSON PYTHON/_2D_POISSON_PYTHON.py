#=========================================================
# NUMERICAL ANALYSIS: USING MESHFREE METHOD

# USING THE REPRODUCING KERNEL COLLOCATION METHOD TO SOLVE 
# THE 2D POISSONS PROBLEM
#=========================================================

import numpy as np
import matplotlib.pyplot as plt

# Files
import forming_NS_NC

#-------------------------
#INPUT PARAMETERS
#-------------------------
showPlot = True #Plotting is done if true
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

NC = forming_NS_NC.forming_CollocationPts(xdim1, xdim2, ydim1, ydim2, CP_x, CP_y)






#-------------------------
#PLOTTING
#-------------------------
if showPlot == True:

    plt.scatter(NS[:,0], NS[:,1], c="r", marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.title("Source Points")
    plt.show()




