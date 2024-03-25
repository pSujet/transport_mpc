import numpy as np
from transporter.utils import function 
import casadi as ca
import matplotlib.pyplot as plt

class PathGenerator:
    '''
    Class to generate paths for the transporter
    '''
    def __init__(self,model) -> None:
        self.model = model

    def generate_circle(self,center,radius,points,loop_time = 60):
        '''
        Generate a circular path
        '''
        # -- generate circle
        points = points
        alpha = np.linspace(0,2*np.pi,points)
        x = center[0] + radius*np.cos(alpha)
        y = center[1] + radius*np.sin(alpha)   

        alpha_dot = 2*np.pi/loop_time                # constant
        x_dot = -radius*alpha_dot*np.sin(alpha)
        y_dot = radius*alpha_dot*np.cos(alpha)

        # -- create reference points array
        xref_array = np.zeros([18,points])
        for point in range(points):
            pL = ca.DM([[x[point]],[y[point]]])
            theL = ca.DM([alpha[point]])
            vL = ca.DM([[x_dot[point]],[y_dot[point]]])
            wL = alpha_dot
            p1,p2,the1,the2 = self.model.get_equi_pos_2D(pL,theL)
            the_vec = ca.DM([np.cos(theL),np.sin(theL)])
            v1 = vL - alpha_dot*0.9*the_vec
            v2 = vL + alpha_dot*0.9*the_vec
            w1 = wL
            w2 = wL
            xref_array[:,[point]] = ca.vertcat(p1,v1,the1,w1,p2,v2,the2,w2,pL,vL,theL,wL)

        self.xref_array = xref_array

        return xref_array
    
    def generate_eight_figure(self,a,points,loop_time = 60):
        '''
        Generate a figure eight path
        '''
        # -- generate eight figure
        points = points
        alpha = np.linspace(0,2*np.pi,points)
        a = a
        x = a*np.sin(alpha)
        y = a*np.sin(alpha)*np.cos(alpha)

        alpha_dot = 2*np.pi/loop_time                # constant
        x_dot = a*np.cos(alpha)*alpha_dot
        y_dot = a*np.cos(alpha)*np.cos(alpha)*alpha_dot - a*np.sin(alpha)*np.sin(alpha)*alpha_dot

        # -- create reference points array
        xref_array = np.zeros([18,points])
        for point in range(points):
            pL = ca.DM([[x[point]],[y[point]]])
            theL = ca.DM([alpha[point]])
            # xref_array[:,[point]] = ca.vertcat(p1,0,0,the1,0,p2,0,0,the2,0,pL,0,0,theL,0)
            vL = ca.DM([[x_dot[point]],[y_dot[point]]])
            wL = alpha_dot
            p1,p2,the1,the2 = self.model.get_equi_pos_2D(pL,theL)
            the_vec = ca.DM([np.cos(theL),np.sin(theL)])
            v1 = vL - alpha_dot*0.9*the_vec
            v2 = vL + alpha_dot*0.9*the_vec
            w1 = wL
            w2 = wL
            xref_array[:,[point]] = ca.vertcat(p1,v1,the1,w1,p2,v2,the2,w2,pL,vL,theL,wL)
        
        self.xref_array = xref_array

        return xref_array
    
    def show_path(self):
        '''
        Show the path generated
        '''
        fig, ax = plt.subplots(1,1)
        ax.plot(self.xref_array[12,:],self.xref_array[13,:],label='path', color = 'k', linestyle = '--',linewidth = 3, marker = 'x')
        ax.set_ylabel("y [m]")
        ax.set_xlabel("x [m]")
        ax.set_aspect('equal')
        ax.grid()
        # ax.set_xlim([-1.0,0.0])
        # ax.set_ylim([-0.5,0.5])
        ax.legend()
    
    def show_state(self):
        '''
        Show the path generated
        '''
        fig, ax = plt.subplots(1,1)
        ax.plot(self.xref_array[12,:],self.xref_array[13,:],label='path', color = 'k', linestyle = '--',linewidth = 2, marker = 'x')
        ax.plot(self.xref_array[0,:],self.xref_array[1,:],label='path', color = 'c', linestyle = '--',linewidth = 2, marker = 'x')
        # ax.plot(self.xref_array[6,:],self.xref_array[7,:],label='path', color = 'y', linestyle = '--',linewidth = 2, marker = 'x')
        for point_index in range(self.xref_array[12,:].shape[0]):
            point_x = self.xref_array[12,point_index]
            point_y = self.xref_array[13,point_index]
            v_x = self.xref_array[14,point_index]
            v_y = self.xref_array[15,point_index]
            robot1_x = self.xref_array[0,point_index]
            robot1_y = self.xref_array[1,point_index]
            robot1_vx = self.xref_array[2,point_index]
            robot1_vy = self.xref_array[3,point_index]

            robot2_x = self.xref_array[6,point_index]
            robot2_y = self.xref_array[7,point_index]
            robot2_vx = self.xref_array[8,point_index]
            robot2_vy = self.xref_array[9,point_index]

            ax.plot([point_x,point_x+v_x],[point_y,point_y+v_y], color = 'g', linestyle = '-',linewidth = 2)
            ax.plot([robot1_x,robot1_x+robot1_vx],[robot1_y,robot1_y+robot1_vy], color = 'r', linestyle = '-',linewidth = 2)
            # ax.plot([robot2_x,robot2_x+robot2_vx],[robot2_y,robot2_y+robot2_vy], color = 'b', linestyle = '-',linewidth = 2)
        ax.set_ylabel("y [m]")
        ax.set_xlabel("x [m]")
        ax.set_aspect('equal')
        ax.grid()
        # ax.set_xlim([-1.0,0.0])
        # ax.set_ylim([-0.5,0.5])
        ax.legend()



