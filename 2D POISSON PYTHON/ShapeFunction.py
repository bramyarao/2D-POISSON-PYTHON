# ============================================================
# Contains all functions related to Shape Function evaluation
# ============================================================

import numpy as np
import math

# Getting nodes in the SF support
def required_nodes(x,y,NS,ss):
    # Required nodes for constructing the moment matrix is P
    # Selecting the nodes around xI, till 4 times the ss
    P = []

    for interger1 in range(NS.shape[0]):
        x_m = NS[interger1,0]
        y_m = NS[interger1,1]
        if ((math.sqrt((x-x_m)**2 + (y-y_m)**2))<=ss):
            P.append([x_m,y_m])
        else:
            continue    
    
    P = np.array(P)
    return P


#================= MAIN FUNCTIONS ==============================
def SF_2D(x,y,NS,P,ss):
    # this function gives Shape function at any point (x,y) for node (xI,yI)
    # For linear basis we need 3x3 M matrix and for quadratic basis we need 6x6
    # M matrix in 2D    

    P = np.transpose(P)
    P_len = P.shape[1] # Num of cols here after transposing

    M = np.zeros((6, 6)) # basis = 2

    # Evaluation of the Moment matrix, here we take the node positions
    # as xxI and yyI for the summation process to be carried out easily

    for integer_1 in range(P_len):
        xxI = P[0,integer_1]
        yyI = P[1,integer_1]

        HB = np.array([[1,x-xxI, y-yyI, (x-xxI)**2, (y-yyI)**2, (x-xxI)*(y-yyI)]]) 
        zz = (math.sqrt((x-xxI)**2+(y-yyI)**2))/ss
        #phy = PHI is the weight function
        phi = phi_eval(zz)
        M = np.add(M, phi*np.matmul(np.transpose(HB),HB))

    # After we get the Moment matrix we construct the SF
    si = np.zeros((1,NS.shape[0]))

    for int_1 in range(NS.shape[0]):
        xI = NS[int_1,0]
        yI = NS[int_1,1]

        Ho = np.array([[1 ,0,0,0,0,0]])
        H = np.array([[1 ,x-xI, y-yI, (x-xI)**2, (y-yI)**2, (x-xI)*(y-yI)]])

        z = (math.sqrt((x-xI)**2+(y-yI)**2))/ss
        PHI = phi_eval(z)   

        si[0,int_1] = si[0,int_1] + np.matmul(np.matmul(Ho , np.linalg.inv(M)) , np.transpose(H)) * PHI

    return si


def DSF_xx(x,y,NS,P,ss):
    #this function gives the second derivative of the Shape function w.r.t 'x' 
    # at any point (x,y) centered at node (xI,yI)
    # For linear basis we need 3x3 M matrix and for quadratic basis we need 6x6
    # M matrix in 2D

    P = np.transpose(P)
    P_len = P.shape[1] # Num of cols here after transposing

    # Evaluation of derivatives of M: M,x and M,xx
    M = np.zeros((6, 6)) # basis = 2
    M_x = np.zeros((6, 6))
    M_xx = np.zeros((6, 6))

    # Evaluation of the Moment matrix and its derivatives (M, M_x, M_xx), here
    # we take the node positions
    # as xxI and yyI for the summation process to be carried out easily

    for integer_2 in range(P_len):
        xxI = P[0,integer_2]
        yyI = P[1,integer_2]

        h = np.transpose(np.array([[1 ,x-xxI, y-yyI,(x-xxI)**2, (y-yyI)**2, (x-xxI)*(y-yyI)]]))
        h_x = np.transpose(np.array([[0, 1, 0, (2*(x-xxI)), 0, y-yyI]]))
        h_xx = np.transpose(np.array([[0 , 0, 0, 2 ,0,0]]))

        # For FINDING PHI
        zz = (math.sqrt((x-xxI)**2+(y-yyI)**2))/ss
        phi = phi_eval(zz)
        dphi = dphi_eval_x(x,y,xxI,yyI,zz,ss)
        ddphi = ddphi_eval_x(x,y,xxI,yyI,zz,ss)

        M = np.add(M, phi*np.matmul(h, np.transpose(h)))

        temp1 = np.matmul(h_x,np.transpose(h))  *phi
        temp2 = np.matmul(h,  np.transpose(h_x))*phi
        temp3 = np.matmul(h,  np.transpose(h))  *dphi
        temp4 = np.add(temp1 , temp2)
        temp5 = np.add(temp4 , temp3)
        M_x = np.add(M_x , temp5)

        temp1 = np.matmul(h_xx,np.transpose(h))   *phi
        temp2 = np.matmul(h_x ,np.transpose(h_x)) *phi
        temp3 = np.matmul(h_x ,np.transpose(h))   *dphi
        temp4 = np.matmul(h_x ,np.transpose(h_x)) *phi
        temp5 = np.matmul(h   ,np.transpose(h_xx))*phi
        temp6 = np.matmul(h   ,np.transpose(h_x)) *dphi
        temp7 = np.matmul(h_x ,np.transpose(h))   *dphi
        temp8 = np.matmul(h   ,np.transpose(h_x)) *dphi
        temp9 = np.matmul(h   ,np.transpose(h))   *ddphi

        temp10 = np.add(temp1 , temp2)
        temp11 = np.add(temp10 , temp3)
        temp12 = np.add(temp11 , temp4)
        temp13 = np.add(temp12 , temp5)
        temp14 = np.add(temp13 , temp6)
        temp15 = np.add(temp14 , temp7)
        temp16 = np.add(temp15 , temp8)
        temp17 = np.add(temp16 , temp9)
        M_xx = np.add(M_xx,temp17)

    # --------------------------------------------------
    # Evaluating inv(M),x i.e derivative of inv of M
    InvM_x = -1.0*  np.matmul( np.matmul(np.linalg.inv(M),M_x) , np.linalg.inv(M) )

    temp1 = np.matmul(M_xx,np.linalg.inv(M))
    temp2 = 2.0*np.matmul(M_x,InvM_x)
    temp3 = np.add(temp1,temp2)
    InvM_xx = -1.0*np.matmul(np.linalg.inv(M),temp3)

    # --------------------------------------------------
    # MAIN matrices for computing Derivative of Shape function

    SIxx = np.zeros((1,NS.shape[0]))

    for int_1 in range(NS.shape[0]):
        xI = NS[int_1,0]
        yI = NS[int_1,1]

        Ho = np.transpose(np.array([[1 ,0,0,0,0,0]]))
        H = np.transpose(np.array([[1 ,x-xI, y-yI, (x-xI)**2, (y-yI)**2, (x-xI)*(y-yI)]]))
        H_x = np.transpose(np.array([[0, 1, 0, (2*(x-xI)), 0, y-yI]]))
        H_xx = np.transpose(np.array([[0 , 0, 0, 2 ,0,0]]))

        #For finding PHI
        z = (math.sqrt((x-xI)**2+(y-yI)**2))/ss
        PHI = phi_eval(z)
        DPHI = dphi_eval_x(x,y,xI,yI,z,ss)
        DDPHI = ddphi_eval_x(x,y,xI,yI,z,ss)        

        temp1 = np.matmul(InvM_xx,H)   *PHI
        temp2 = np.matmul(np.linalg.inv(M) ,H_xx)*PHI
        temp3 = np.matmul(np.linalg.inv(M) ,H)   *DDPHI
        temp4 = 2.0*PHI*np.matmul(InvM_x ,H_x)
        temp5 = 2.0*DPHI*np.matmul(InvM_x,H)  
        temp6 = 2.0*DPHI*np.matmul(np.linalg.inv(M),H_x)

        temp7 = np.add(temp1,temp2)
        temp8 = np.add(temp7,temp3)
        temp9 = np.add(temp8,temp4)
        temp10 = np.add(temp9,temp5)
        temp11 = np.add(temp10,temp6)        

        SIxx[0,int_1] = SIxx[0,int_1] + np.matmul(np.transpose(Ho),temp11)

    return SIxx

def DSF_yy(x,y,NS,P,ss):
    # this function gives the second derivative of the Shape function w.r.t 'x' 
    # at any point (x,y) centered at node (xI,yI)
    # For linear basis we need 3x3 M matrix and for quadratic basis we need 6x6
    # M matrix in 2D

    P = np.transpose(P)
    P_len = P.shape[1] # Num of cols here after transposing

    # Evaluation of derivatives of M: M,y and M,yy
    M = np.zeros((6, 6)) # basis = 2
    M_y = np.zeros((6, 6))
    M_yy = np.zeros((6, 6))

    # Evaluation of the Moment matrix, here we take the node positions
    # as xxI and yyI for the summation process to be carried out easily

    for integer_2 in range(P_len):
        xxI = P[0,integer_2]
        yyI = P[1,integer_2]

        h = np.transpose(np.array([[1 ,x-xxI, y-yyI,(x-xxI)**2, (y-yyI)**2, (x-xxI)*(y-yyI)]]))
        h_y = np.transpose(np.array([[0,0,1,0,(2*(y-yyI)),(x-xxI)]]))
        h_yy = np.transpose(np.array([[0,0,0,0,2,0]]))

        # For FINDING PHI
        zz = (math.sqrt((x-xxI)**2+(y-yyI)**2))/ss 
        
        phi = phi_eval(zz)
        dphi = dphi_eval_y(x,y,xxI,yyI,zz,ss)
        ddphi = ddphi_eval_y(x,y,xxI,yyI,zz,ss)  

        M = np.add(M, phi*np.matmul(h, np.transpose(h)))

        temp1 = np.matmul(h_y,np.transpose(h))  *phi
        temp2 = np.matmul(h,  np.transpose(h_y))*phi
        temp3 = np.matmul(h,  np.transpose(h))  *dphi
        temp4 = np.add(temp1 , temp2)
        temp5 = np.add(temp4 , temp3)
        M_y = np.add(M_y , temp5)

        temp1 = np.matmul(h_yy,np.transpose(h))   *phi
        temp2 = np.matmul(h_y ,np.transpose(h_y)) *phi
        temp3 = np.matmul(h_y ,np.transpose(h))   *dphi
        temp4 = np.matmul(h_y ,np.transpose(h_y)) *phi
        temp5 = np.matmul(h   ,np.transpose(h_yy))*phi
        temp6 = np.matmul(h   ,np.transpose(h_y)) *dphi
        temp7 = np.matmul(h_y ,np.transpose(h))   *dphi
        temp8 = np.matmul(h   ,np.transpose(h_y)) *dphi
        temp9 = np.matmul(h   ,np.transpose(h))   *ddphi

        temp10 = np.add(temp1 , temp2)
        temp11 = np.add(temp10 , temp3)
        temp12 = np.add(temp11 , temp4)
        temp13 = np.add(temp12 , temp5)
        temp14 = np.add(temp13 , temp6)
        temp15 = np.add(temp14 , temp7)
        temp16 = np.add(temp15 , temp8)
        temp17 = np.add(temp16 , temp9)
        M_yy = np.add(M_yy,temp17)

    # --------------------------------------------------
    # Evaluating inv(M),y i.e derivative of inv of M
    InvM_y = -1.0*  np.matmul( np.matmul(np.linalg.inv(M),M_y) , np.linalg.inv(M) )

    temp1 = np.matmul(M_yy,np.linalg.inv(M))
    temp2 = 2.0*np.matmul(M_y,InvM_y)
    temp3 = np.add(temp1,temp2)
    InvM_yy = -1.0*np.matmul(np.linalg.inv(M),temp3)

    # --------------------------------------------------
    # MAIN matrices for computing Derivative of Shape function
    SIyy = np.zeros((1,NS.shape[0]))

    for int_1 in range(NS.shape[0]):
        xI = NS[int_1,0]
        yI = NS[int_1,1]

        Ho = np.transpose(np.array([[1 ,0,0,0,0,0]]))
        H = np.transpose(np.array([[1 ,x-xI, y-yI, (x-xI)**2, (y-yI)**2, (x-xI)*(y-yI)]]))
        H_y = np.transpose(np.array([[0,0,1,0,(2*(y-yI)),(x-xI)]]))
        H_yy = np.transpose(np.array([[0,0,0,0,2,0]]))

        # For finding PHI
        z = (math.sqrt((x-xI)**2+(y-yI)**2))/ss

        PHI = phi_eval(z)
        DPHI = dphi_eval_y(x,y,xI,yI,z,ss)
        DDPHI = ddphi_eval_y(x,y,xI,yI,z,ss)

        temp1 = np.matmul(InvM_yy,H)   *PHI
        temp2 = np.matmul(np.linalg.inv(M) ,H_yy)*PHI
        temp3 = np.matmul(np.linalg.inv(M) ,H)   *DDPHI
        temp4 = 2.0*PHI*np.matmul(InvM_y ,H_y)
        temp5 = 2.0*DPHI*np.matmul(InvM_y,H)  
        temp6 = 2.0*DPHI*np.matmul(np.linalg.inv(M),H_y)

        temp7 = np.add(temp1,temp2)
        temp8 = np.add(temp7,temp3)
        temp9 = np.add(temp8,temp4)
        temp10 = np.add(temp9,temp5)
        temp11 = np.add(temp10,temp6)        

        SIyy[0,int_1] = SIyy[0,int_1] + np.matmul(np.transpose(Ho),temp11)

    return SIyy



#================= SUB FUNCTIONS ==============================

# Phi
def phi_eval(zz):
    #USING Quintic B-SPLINE
    if zz >= 0 and zz<(1/3):
        phi = (11/20)-(9/2)*zz**2 + (81/4)*zz**4 -(81/4)*zz**5
    elif zz>=1/3 and zz<2/3:
        phi = (17/40) + (15/8)*zz - (63/4)*zz**2 + (135/4)*zz**3 - (243/8)*zz**4 + (81/8)*zz**5
    elif zz>=2/3 and zz<1:
        phi = (81/40) - (81/8)*zz + (81/4)*zz**2 - (81/4)*zz**3 + (81/8)*zz**4 - (81/40)*zz**5
    elif zz>=1:
        phi = 0
    return phi


# Phi 1st derivative wrt 'x'
def dphi_eval_x(x,y,xxI,yyI,zz,ss):
    if zz>=0 and zz<(1/3):
        dphi = (81*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2))/(2*ss**4) - (((405*x)/4 - (405*xxI)/4)*((x - xxI)**2 + (y - yyI)**2)**(3/2))/ss**5 - (9*x - 9*xxI)/ss**2
    elif zz>=1/3 and zz<2/3:
        dphi = (15*(2*x - 2*xxI))/(16*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) - (63*(2*x - 2*xxI))/(4*ss**2) + (405*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(8*ss**3) + (405*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(16*ss**5) - (243*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2))/(4*ss**4)
    elif zz>=2/3 and zz<1:
        dphi = (81*(2*x - 2*xxI))/(4*ss**2) - (81*(2*x - 2*xxI))/(16*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) - (243*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(8*ss**3) - (81*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(16*ss**5) + (81*(2*x - 2*xxI)*((x - xxI)**2 + (y - yyI)**2))/(4*ss**4)
    elif zz>=1:
        dphi = 0
    return dphi


# Phi 2nd derivative wrt 'x'
def ddphi_eval_x(x,y,xxI,yyI,zz,ss):
    if zz>=0 and zz<(1/3):
        ddphi = (81*(2*x - 2*xxI)**2)/(2*ss**4) - (405*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(4*ss**5) + (81*((x - xxI)**2 + (y - yyI)**2))/ss**4 - 9/ss**2 - (1215*(2*x - 2*xxI)**2*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(16*ss**5)
    elif zz>=1/3 and zz<2/3:
        ddphi = 15/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) + (405*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(4*ss**3) + (405*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(8*ss**5) - (243*(x - xxI)**2)/ss**4 - (243*(x**2 - 2*x*xxI + xxI**2 + y**2 - 2*y*yyI + yyI**2))/(2*ss**4) - 63/(2*ss**2) - (15*(x - xxI)**2)/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(3/2)) + (405*(x - xxI)**2)/(4*ss**3*((x - xxI)**2 + (y - yyI)**2)**(1/2)) + (1215*((x - xxI)**2 + (y - yyI)**2)**(1/2)*(x - xxI)**2)/(8*ss**5)
    elif zz>=2/3 and zz<1:
        ddphi = (81*(x - xxI)**2)/ss**4 - (243*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(4*ss**3) - (81*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(8*ss**5) - 81/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) + (81*(x**2 - 2*x*xxI + xxI**2 + y**2 - 2*y*yyI + yyI**2))/(2*ss**4) + 81/(2*ss**2) + (81*(x - xxI)**2)/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(3/2)) - (243*(x - xxI)**2)/(4*ss**3*((x - xxI)**2 + (y - yyI)**2)**(1/2)) - (243*((x - xxI)**2 + (y - yyI)**2)**(1/2)*(x - xxI)**2)/(8*ss**5)
    elif zz>=1:
        ddphi = 0
    return ddphi


# Phi 1st derivative wrt 'y'
def dphi_eval_y(x,y,xxI,yyI,zz,ss):
    if zz>=0 and zz<(1/3):
        dphi = (81*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2))/(2*ss**4) - (((405*y)/4 - (405*yyI)/4)*((x - xxI)**2 + (y - yyI)**2)**(3/2))/ss**5 - (9*y - 9*yyI)/ss**2
    elif zz>=1/3 and zz<2/3:
        dphi = (15*(2*y - 2*yyI))/(16*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) - (63*(2*y - 2*yyI))/(4*ss**2) + (405*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(8*ss**3) + (405*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(16*ss**5) - (243*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2))/(4*ss**4)
    elif zz>=2/3 and zz<1:
        dphi = (81*(2*y - 2*yyI))/(4*ss**2) - (81*(2*y - 2*yyI))/(16*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) - (243*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(8*ss**3) - (81*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(16*ss**5) + (81*(2*y - 2*yyI)*((x - xxI)**2 + (y - yyI)**2))/(4*ss**4)
    elif zz>=1:
        dphi = 0
    return dphi


# Phi 2nd derivative wrt 'y'
def  ddphi_eval_y(x,y,xxI,yyI,zz,ss):
    if zz>=0 and zz<(1/3):
        ddphi = (81*(2*y - 2*yyI)**2)/(2*ss**4) - (405*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(4*ss**5) + (81*((x - xxI)**2 + (y - yyI)**2))/ss**4 - 9/ss**2 - (1215*(2*y - 2*yyI)**2*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(16*ss**5);
    elif zz>=1/3 and zz<2/3:
        ddphi = 15/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) + (405*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(4*ss**3) + (405*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(8*ss**5) - (243*(y - yyI)**2)/ss**4 - (243*(x**2 - 2*x*xxI + xxI**2 + y**2 - 2*y*yyI + yyI**2))/(2*ss**4) - 63/(2*ss**2) - (15*(y - yyI)**2)/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(3/2)) + (405*(y - yyI)**2)/(4*ss**3*((x - xxI)**2 + (y - yyI)**2)**(1/2)) + (1215*((x - xxI)**2 + (y - yyI)**2)**(1/2)*(y - yyI)**2)/(8*ss**5);
    elif zz>=2/3 and zz<1:
        ddphi = (81*(y - yyI)**2)/ss**4 - (243*((x - xxI)**2 + (y - yyI)**2)**(1/2))/(4*ss**3) - (81*((x - xxI)**2 + (y - yyI)**2)**(3/2))/(8*ss**5) - 81/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(1/2)) + (81*(x**2 - 2*x*xxI + xxI**2 + y**2 - 2*y*yyI + yyI**2))/(2*ss**4) + 81/(2*ss**2) + (81*(y - yyI)**2)/(8*ss*((x - xxI)**2 + (y - yyI)**2)**(3/2)) - (243*(y - yyI)**2)/(4*ss**3*((x - xxI)**2 + (y - yyI)**2)**(1/2)) - (243*((x - xxI)**2 + (y - yyI)**2)**(1/2)*(y - yyI)**2)/(8*ss**5);
    elif zz>=1:
        ddphi = 0;
    return ddphi




