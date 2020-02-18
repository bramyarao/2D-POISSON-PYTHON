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
import ShapeFunction

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

num_pts = np.array([10,10,40,40], float)
#num_pts = np.array([5,5,5,5], float)
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
A2 = forming_A.part_of_NEB(NEB,NS,ss,sq_alphag)

A = np.concatenate((A1,A2))


#-------------------------
# Forming b vector 
#-------------------------

b = np.zeros((no_NC+no_NEB,1))
no_int = NC.shape[0]
int_1 = 0

# INTERIOR force term
for int_2 in range(no_int):
    xtemp = NC[int_2,0]
    ytemp = NC[int_2,1]        
    b[int_1] = b[int_1] + (xtemp**2 + ytemp**2)*math.exp(xtemp*ytemp)       
    int_1 += 1

# int_1 is getting incremented
# EB force terms

for int_2 in range(NEB.shape[0]):
    xtemp = NEB[int_2,0]
    ytemp = NEB[int_2,1]        
    b[int_1] = b[int_1] + sq_alphag*math.exp(xtemp*ytemp)         
    int_1 += 1

#-------------------------
# Solving the system
#------------------------

AT = np.transpose(A)
Afinal = np.matmul(AT,A)
bfinal = np.matmul(AT,b)

a = np.linalg.solve(Afinal,bfinal) # Least squares solution ,rcond=None

#-----------------------------------------------------------------------
# Comparing Solutions Along the Diagonal through (0,0) & (1,1)
#-----------------------------------------------------------------------

x_con = np.linspace(xdim1,xdim2,20)
y_con = np.linspace(ydim1,ydim2,20)

u_con = np.zeros((len(x_con),1))
u_exact_con = np.zeros((len(x_con),1))

for int1 in range(len(x_con)):
    x = x_con[int1]
    y = y_con[int1]
    P = ShapeFunction.required_nodes(x,y,NS,ss)
    SI = ShapeFunction.SF_2D(x,y,NS,P,ss)

    # for getting uh value of u_approx at each of the points (x,y)
    u_con[int1,0] = np.matmul(SI,a)

    # Finding u_exact at the point (x,y)
    u_exact_con[int1,0] = math.exp(x*y)

# Diagonal length
DL = np.zeros((len(x_con),1))
for int6 in range(len(x_con)):
    DL[int6,0]  = math.sqrt((x_con[int6])**2 + (y_con[int6])**2)

#-----------------------------------------------------------------------
# Plotting the numerical solution
#-----------------------------------------------------------------------
uScatter = np.zeros((NC.shape[0],1))
uScatter_exact = np.zeros((NC.shape[0],1))

for int1 in range(NC.shape[0]):
    x = NC[int1,0]
    y = NC[int1,1]
    P = ShapeFunction.required_nodes(x,y,NS,ss)
    SI = ShapeFunction.SF_2D(x,y,NS,P,ss)
    
    # for getting uh value of u_approx at each of the points (x,y)
    uScatter[int1,0] = np.matmul(SI,a)
        
    # Finding u_exact at the point (x,y)
    uScatter_exact[int1,0] = math.exp(x*y)

#-------------------------
# PRINTING
#-------------------------
if (printStatements == True):
    print('Source points %d' %NS.shape[0])
    print('Collocation points %d' %NC.shape[0])
    print('Interior collocation points %d' %NI_c.shape[0])
    print('EB collocation points %d' %NEB.shape[0])


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
    plt.show()

    plt.figure(3)
    plt.plot(DL, u_con, 'ko', label='RKCM')
    plt.plot(DL, u_exact_con, 'k--', label='Analytical')
    plt.xlabel("Diagonal length")
    plt.ylabel("u(x,y)")
    plt.xlim(0.0,1.5)
    plt.ylim(0.9,3.0)
    plt.legend()
    plt.show()

    plt.figure(4)
    fig4=plt.scatter(NC[:,0], NC[:,1], c=uScatter[:,0],cmap='hsv')
    plt.colorbar(fig4)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u(x,y) Numerical (RKCM)")
    plt.show()

    plt.figure(5)
    fig5=plt.scatter(NC[:,0], NC[:,1], c=uScatter_exact[:,0],cmap='hsv')
    plt.colorbar(fig5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u(x,y) Analytical (Exact)")
    interactive(False)
    plt.show()


    # For Saving: plt.savefig('ScatterPlot.png')
    # To show all figures at a time:
    # - only first and last figures use interactive, middle ones do not
    # - also first figure should have interactive(True)
    # - and last one should have interactive(False)






