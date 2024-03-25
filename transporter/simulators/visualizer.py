'''
Helper visualization functions for plotting
'''
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from transporter.utils.function import *

def plot_robot(p,q,ax,clr = 'g',cle = 'darkred', alpha = 1):   
    xc = p[0]
    yc = p[1]
    zc = p[2]

    # --- Sphere
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    r = 0.5#0.2#0.5
    x = r * np.outer(np.cos(u), np.sin(v)) + xc
    y = r * np.outer(np.sin(u), np.sin(v)) + yc
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zc
    ax.plot_surface(x, y, z, alpha=0.2*alpha*2, color=clr,linewidth=1, edgecolors=(0, 1, 0, 0.0))

    ax.plot3D(xc,yc,zc, marker="o", markersize=10, markeredgewidth=2, markeredgecolor=clr, markerfacecolor="white", alpha = alpha)
    ax.plot3D(xc,yc,zc, marker="o", markersize=5, markeredgewidth=2, markeredgecolor="black", markerfacecolor="white", alpha = alpha)
    plot_axis(xc,yc,zc,q,ax, alpha = alpha)

def plot_load(p,q,ax,clr = 'g', alpha = 1):   
    xc = p[0]
    yc = p[1]
    zc = p[2]
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    r = 1#0.2 #1
    x = r * np.outer(np.cos(u), np.sin(v)) + xc
    y = r * np.outer(np.sin(u), np.sin(v)) + yc
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zc
    ax.plot_surface(x, y, z, alpha=0.2*alpha*2, color=clr,linewidth=1, edgecolors=(0, 1, 0, 0.0))
    ax.plot3D(xc,yc,zc, marker="o", markersize=10, markeredgewidth=2, markeredgecolor="darkgreen", markerfacecolor="white", alpha = alpha)
    ax.plot3D(xc,yc,zc, marker="o", markersize=5, markeredgewidth=2, markeredgecolor="black", markerfacecolor="white", alpha = alpha)
    plot_axis(xc,yc,zc,q,ax, alpha = alpha)

def plot_cable(p,s,ax, alpha = 1):   
    lx = [p[0,0],s[0,0]]
    ly = [p[1,0],s[1,0]]
    lz = [p[2,0],s[2,0]]
    ax.plot3D(p[0,0], p[1,0], p[2,0], marker="o", markersize=3, markeredgewidth=2, markeredgecolor="black", markerfacecolor="white", alpha = alpha)
    ax.plot3D(s[0,0], s[1,0], s[2,0], marker="o", markersize=3, markeredgewidth=2, markeredgecolor="black", markerfacecolor="white", alpha = alpha)
    ax.plot3D(lx,ly,lz,linewidth=3, color='k', alpha = alpha)

def plot_vector(p,s,ax,clr = 'm',scale = 1.5):   
    size = np.linalg.norm(s)
    ax.quiver(p[0],p[1],p[2],s[0,0],s[1,0],s[2,0], length=size*scale, normalize=True, color=clr,linewidth=2)

def plot_path(xref,ax,alpha = 0.2,PLANAR=False,THREE = False):
    '''
    Plot reference path
    '''
    if PLANAR:
        x = np.array(xref[12]).reshape(-1)
        y = np.array(xref[13]).reshape(-1)
        z = np.array([0]).reshape(-1)
        q = np.array(angle2quatZ(xref[16])).reshape(-1)
    elif THREE:
        x = np.array(xref[39]).reshape(-1)
        y = np.array(xref[40]).reshape(-1)
        z = np.array(xref[41]).reshape(-1)
        q = np.array(xref[45:49])
    else:
        x = np.array(xref[26]).reshape(-1)
        y = np.array(xref[27]).reshape(-1)
        z = np.array(xref[28]).reshape(-1)
        q = np.array(xref[32:36])
    ax.plot3D(x,y,z, marker="x", markersize=10, markeredgewidth=4, markeredgecolor="black",alpha = alpha)
    plot_axis(x,y,z,q,ax, alpha = alpha)

def plot_axis(xc,yc,zc,q,ax,alpha=1,size = 0.4):
    x_axis = np.array(dcm_e2b(q).T @ np.array([[1],[0],[0]]))*size
    y_axis = np.array(dcm_e2b(q).T @ np.array([[0],[1],[0]]))*size
    z_axis = np.array(dcm_e2b(q).T @ np.array([[0],[0],[1]]))*size
    x_axisx = np.array([xc, xc + x_axis[0][0]]).reshape(-1)
    x_axisy = np.array([yc, yc + x_axis[1][0]]).reshape(-1)
    x_axisz = np.array([zc, zc + x_axis[2][0]]).reshape(-1)
    y_axisx = np.array([xc, xc + y_axis[0][0]]).reshape(-1)
    y_axisy = np.array([yc, yc + y_axis[1][0]]).reshape(-1)
    y_axisz = np.array([zc, zc + y_axis[2][0]]).reshape(-1)
    z_axisx = np.array([xc, xc + z_axis[0][0]]).reshape(-1)
    z_axisy = np.array([yc, yc + z_axis[1][0]]).reshape(-1)
    z_axisz = np.array([zc, zc + z_axis[2][0]]).reshape(-1)
    ax.plot3D(x_axisx,x_axisy,x_axisz,linewidth=3, color='r', alpha = alpha)
    ax.plot3D(y_axisx,y_axisy,y_axisz,linewidth=3, color='g', alpha = alpha)
    ax.plot3D(z_axisx,z_axisy,z_axisz,linewidth=3, color='b', alpha = alpha)

def plot_system(ax,i,x_Log,model,alpha = 1, PLANAR = False, THREE = False):
    if PLANAR:
        # == Extract variable for visualization    
        # -- states robot 1
        p1_vis = np.vstack([x_Log[0:2],0])
        v1_vis = np.vstack([x_Log[2:4],0]).reshape((-1, 1))
        the1_vis = x_Log[4]
        q1_vis = angle2quatZ(the1_vis)
        w1_vis = np.vstack([0,0,x_Log[5]])            

        # -- states robot 2
        p2_vis = np.vstack([x_Log[6:8],0])
        v2_vis = np.vstack([x_Log[8:10],0]).reshape((-1, 1))
        the2_vis = x_Log[10]
        q2_vis = angle2quatZ(the2_vis)
        w2_vis = np.vstack([0,0,x_Log[11]]) 

        # -- states Load
        pL_vis = np.vstack([x_Log[12:14],0])
        vL_vis = np.vstack([x_Log[14:16],0]).reshape((-1, 1))
        theL_vis = x_Log[16]
        qL_vis = angle2quatZ(theL_vis)
        wL_vis = np.vstack([0,0,x_Log[17]]) 

        # -- states connection
        p1_prime_vis = np.array(p1_vis + dcm_e2b(q1_vis).T @ ca.vertcat(model.r1,0))
        s1_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ ca.vertcat(model.R1,0))
        p2_prime_vis = np.array(p2_vis + dcm_e2b(q2_vis).T @ ca.vertcat(model.r2,0))
        s2_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ ca.vertcat(model.R2,0))

        # -- visualization
        plot_robot(p1_vis,q1_vis,ax,clr='red',cle='darkred', alpha = alpha)
        plot_robot(p2_vis,q2_vis,ax,clr='blue',cle='darkblue', alpha = alpha)
        plot_load(pL_vis,qL_vis,ax, alpha = alpha)
        plot_cable(p1_prime_vis,s1_vis,ax, alpha = alpha)
        plot_cable(p2_prime_vis,s2_vis,ax, alpha = alpha)
    
    elif THREE:
        # == Extract variable for visualization    
        # -- states robot 1
        p1_vis = x_Log[0:3]
        v1_vis = x_Log[3:6].reshape((-1, 1))
        q1_vis = x_Log[6:10]
        w1_vis = x_Log[10:13]            

        # -- states robot 2
        p2_vis = x_Log[13:16]
        v2_vis = x_Log[16:19].reshape((-1, 1))
        q2_vis = x_Log[19:23]
        w2_vis = x_Log[23:26]

        # -- states robot 3
        p3_vis = x_Log[26:29]
        v3_vis = x_Log[29:32].reshape((-1, 1))
        q3_vis = x_Log[32:36]
        w3_vis = x_Log[36:39]

        # -- states Load
        pL_vis = x_Log[39:42]
        vL_vis = x_Log[42:45].reshape((-1, 1))
        qL_vis = x_Log[45:49]
        wL_vis = x_Log[49:52]

        # -- states connection
        p1_prime_vis = np.array(p1_vis + dcm_e2b(q1_vis).T @ model.r1)
        s1_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ model.R1)
        p2_prime_vis = np.array(p2_vis + dcm_e2b(q2_vis).T @ model.r2)
        s2_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ model.R2)
        p3_prime_vis = np.array(p3_vis + dcm_e2b(q3_vis).T @ model.r3)
        s3_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ model.R3)

        # -- agents
        plot_robot(p1_vis,q1_vis,ax,clr='red',cle='darkred', alpha = alpha)
        plot_robot(p2_vis,q2_vis,ax,clr='blue',cle='darkblue', alpha = alpha)
        plot_robot(p3_vis,q3_vis,ax,clr='violet',cle='darkviolet', alpha = alpha)
        plot_load(pL_vis,qL_vis,ax, alpha = alpha)
        plot_cable(p1_prime_vis,s1_vis,ax, alpha = alpha)
        plot_cable(p2_prime_vis,s2_vis,ax, alpha = alpha)
        plot_cable(p3_prime_vis,s3_vis,ax, alpha = alpha)

    else:
        # == Extract variable for visualization    
        # -- states robot 1
        p1_vis = x_Log[0:3]
        v1_vis = x_Log[3:6].reshape((-1, 1))
        q1_vis = x_Log[6:10]
        w1_vis = x_Log[10:13]            

        # -- states robot 2
        p2_vis = x_Log[13:16]
        v2_vis = x_Log[16:19].reshape((-1, 1))
        q2_vis = x_Log[19:23]
        w2_vis = x_Log[23:26]

        # -- states Load
        pL_vis = x_Log[26:29]
        vL_vis = x_Log[29:32].reshape((-1, 1))
        qL_vis = x_Log[32:36]
        wL_vis = x_Log[36:39]

        # -- states connection
        p1_prime_vis = np.array(p1_vis + dcm_e2b(q1_vis).T @ model.r1)
        s1_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ model.R1)
        p2_prime_vis = np.array(p2_vis + dcm_e2b(q2_vis).T @ model.r2)
        s2_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ model.R2)

        plot_robot(p1_vis,q1_vis,ax,clr='red',cle='darkred', alpha = alpha)
        plot_robot(p2_vis,q2_vis,ax,clr='blue',cle='darkblue', alpha = alpha)
        plot_load(pL_vis,qL_vis,ax, alpha = alpha)
        plot_cable(p1_prime_vis,s1_vis,ax, alpha = alpha)
        plot_cable(p2_prime_vis,s2_vis,ax, alpha = alpha)

def trunc(values, decs=0):
    '''
    For printing proper decimal
    '''
    return np.trunc(values*10**decs)/(10**decs)