'''
Set of utility functions used across the package.
'''

import casadi as ca
import numpy as np

def dcm_e2b(q):
    '''
    Create DCM (direction cosine matrix), a transformation matrix from Earth frame to Body frame
    use Lambda.T to transform from Body to Earth.
    '''
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    Lambda = ca.vertcat(ca.horzcat(q1**2-q2**2-q3**2+q4**2  , 2*(q1*q2+q3*q4)           , 2*(q1*q3-q2*q4)),
                        ca.horzcat(2*(q1*q2-q3*q4)          , -q1**2+q2**2-q3**2+q4**2  , 2*(q2*q3+q1*q4)),
                        ca.horzcat(2*(q1*q3+q2*q4)          , 2*(q2*q3-q1*q4)           , -q1**2-q2**2+q3**2+q4**2))
    return Lambda

def dcm_e2b2D(theta):
    '''
    Create DCM (direction cosine matrix), a transformation matrix from Earth frame to Body frame
    use Lambda.T to transform from Body to Earth in 2D.
    '''
    Lambda = ca.vertcat(ca.horzcat(np.cos(theta)  , np.sin(theta)),
                        ca.horzcat(-np.sin(theta) , np.cos(theta)))
    return Lambda

def angle2quatZ(theta):
    q = np.vstack([0,
                   0,
                   np.sin(theta/2),
                   np.cos(theta/2)])
    return q

def omega_oper(w):
    '''
    Create Omega operator for quaternion derivative
    '''
    wx = w[0]
    wy = w[1]
    wz = w[2]
    Omeg = ca.vertcat(ca.horzcat(0, wz, -wy, wx),
                      ca.horzcat(-wz, 0, wx, wy),
                      ca.horzcat(wy, -wx, 0, wz),
                      ca.horzcat(-wx, -wy, -wz, 0))
    return Omeg

def skew(w):
    '''
    Create skew symmetric matrix for the cross product operation 
    '''
    w1 = w[0]
    w2 = w[1]
    w3 = w[2]
    skew = ca.vertcat(ca.horzcat(0, -w3, w2),
                      ca.horzcat(w3, 0, -w1),
                      ca.horzcat(-w2, w1, 0))
    return skew

def skew2D(w):
    '''
    Create skew symmetric matrix for the cross product operation in 2D
    '''
    skew = ca.vertcat(ca.horzcat(0, -w),
                      ca.horzcat(w, 0))
    return skew

def skewSca2D(r):
    '''
    Create skew symmetric matrix for the cross product operation in 2D time scalar
    '''
    rx = r[0]
    ry = r[1]
    skew = ca.vertcat(ca.horzcat(ry),
                      ca.horzcat(-rx))
    return skew

def skewVec2D(r):
    '''
    Create skew symmetric matrix for the cross product operation in 2D time vector
    '''
    rx = r[0]
    ry = r[1]
    skew = ca.horzcat(-ry,rx)

    return skew

def skewsqr2D(r1,r2):
    '''
    Create skew symmetric matrix for the cross product operation in 2D
    '''
    r1x = r1[0]
    r1y = r1[1]
    r2x = r2[0]
    r2y = r2[1]
    skew = ca.vertcat(ca.horzcat(-r1y*r2y,r1y*r2x),
                      ca.horzcat(-r1x*r2y,r1x*r2x))
    return skew

def get_equi(pL,qL,R1,R2,l1,l2,r1,r2):
    '''
    Get the Equilibrium states (position and rotation) of transporters based on states of the load 
    (1)--(L)--(2)
    '''
    LdL = dcm_e2b(qL)
    s1 = pL + LdL.T @ R1
    s2 = pL + LdL.T @ R2
    b = (s1 - s2)/np.sqrt((s1 - s2).T @ (s1 - s2))
    p1 = s1 + (l1 + np.linalg.norm(r1)) @ b
    p2 = s2 - (l2 + np.linalg.norm(r2)) @ b
    q1 = qL
    q2 = qL
    return p1,p2,q1,q2

def get_equi_2D(pL,theL,R1,R2,l1,l2,r1,r2):
    '''
    Get the Equilibrium states (position and rotation) of transporters based on states of the load 
    (1)--(L)--(2)
    '''
    LdL = dcm_e2b2D(theL)
    s1 = pL + LdL.T @ R1
    s2 = pL + LdL.T @ R2
    b = (s1 - s2)/np.sqrt((s1 - s2).T @ (s1 - s2))
    p1 = s1 + (l1 + np.linalg.norm(r1)) @ b
    p2 = s2 - (l2 + np.linalg.norm(r2)) @ b
    the1 = theL
    the2 = theL
    return p1,p2,the1,the2

def get_equi_3A(pL,qL,R1,R2,R3,l1,l2,l3,r1,r2,r3):
    '''
    Get the Equilibrium states (position and rotation) of transporters based on states of the load 
                     (1)
                      |
                      |
                     (L)
                    /   \
                  /       \ 
                (2)        (3)
    '''
    LdL = dcm_e2b(qL)
    s1 = pL + LdL.T @ R1
    s2 = pL + LdL.T @ R2
    s3 = pL + LdL.T @ R3
    c = (s1+s2+s3)/3
    b1 = (s1 - c)/np.sqrt((s1 - c).T @ (s1 - c)) 
    b2 = (s2 - c)/np.sqrt((s2 - c).T @ (s2 - c)) 
    b3 = (s3 - c)/np.sqrt((s3 - c).T @ (s3 - c)) 
    p1 = s1 + (l1 + np.linalg.norm(r1)) @ b1
    p2 = s2 + (l2 + np.linalg.norm(r2)) @ b2
    p3 = s3 + (l3 + np.linalg.norm(r3)) @ b3
    q1 = qL
    q2 = qL
    q3 = qL
    return p1,p2,p3,q1,q2,q3

def quatconj(q):
    '''
    Create conjute of quaternion
    '''
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    conj = ca.vertcat(-q1,-q2,-q3,q4)
    return conj
    
def quatkroneck(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    kroneck = ca.vertcat(ca.horzcat(-q1, -q2, -q3, q4),
                        ca.horzcat(q4, -q3, q2, q1),
                        ca.horzcat(q3, q4, -q1, q2),
                        ca.horzcat(-q2, q1, q4, q3))
    return kroneck

def quatdist(p,q):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    dist = (1 - (p1*q1 + p2*q2 + p3*q3 + p4*q4)**2)*np.pi#*5
    return dist

def quatdistexplicit(p,q):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    dist = np.arccos(2*(p1*q1 + p2*q2 + p3*q3 + p4*q4)**2-1)
    return dist

def quat2euler(q):
    # in 3-2-1 sequence
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]

    roll = np.arctan2(2*(q4*q1+q2*q3),1-2*(q1**2+q2**2))
    # pitch = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(q4*q2-q1*q3)),np.sqrt(1-2*(q4*q2-q1*q3)))
    pitch = np.arcsin(2*(q4*q2-q1*q3))
    yaw = np.arctan2(2*(q4*q3+q1*q2),1-2*(q2**2+q3**2))

    return np.array([[roll],[pitch],[yaw]])

def yaw2quaternion(yaw):
    """
    Convert yaw (rotation around the vertical axis) to a quaternion.        
    :param yaw: Yaw angle in radians.
    :return: Quaternion.
    """    
    q1 = 0  # No rotation around the x-axis for yaw
    q2 = 0  # No rotation around the y-axis for yaw
    q3 = np.sin(yaw / 2)     
    q4 = np.cos(yaw / 2)   
    return ca.vertcat(q1,q2,q3,q4)



