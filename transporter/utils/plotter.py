'''
Class for plot visualization for 2D case
'''
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transporter.utils.function import *
from transporter.simulators.visualizer import *

class Plotter:
    def __init__(self,data,model) -> None:
        self.data = data
        self.t_sim = self.data['t_sim']
        self.n_sim = self.data['n_sim']
        self.model = model

    def plot_all(self):
        '''
        Plot all the plots
        '''
        self.plot_cost()
        # self.plot_error()        
        self.plot_states_track('load')    
        self.plot_states_track('transporter1') 
        self.plot_states_track('transporter2')
        self.plot_traj()
        self.plot_input('transporter1')    
        self.plot_input('transporter2')  
        # self.plot_tension()
        
        plt.show()

    def plot_cost(self):
        '''
        Plot the MPC cost and computation time over simualtion time
        '''
        cost_Log = self.data['cost']
        time_Log = self.data['compute_time']
        t_cost = np.linspace(0,self.t_sim,self.n_sim)

        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(2,1)
        # fig.suptitle('Cost')

        axs[0].plot(t_cost,cost_Log[0,:],label='cost')
        axs[0].set_ylabel("Running Cost")
        axs[0].set_xlabel("Time [s]")
        axs[0].grid()
        axs[0].set_xlim([0,self.t_sim])

        axs[1].plot(t_cost[1:],time_Log[0,1:],label='time',marker=".")
        axs[1].set_ylabel("Computed Time [s]")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid()
        axs[1].set_xlim([0,self.t_sim])
        axs[1].axhline(y = np.mean(time_Log[0,1:]), color = 'r', linestyle = '--')
        print("Max initialized computed time: " + str(np.max(time_Log))+" s")
        print("Max computed time: " + str(np.max(time_Log[0,1:]))+" s")
        print("Average computed time: "+ str(np.mean(time_Log[0,1:]))+" s")
        fig.tight_layout()
        # plt.show()

    def plot_tension(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim)
        direct_Log = self.data['direct'] 

        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(2,1)
        # fig.suptitle("Cable Tension",fontsize = 16)

        axs[0].plot(t_x,direct_Log[0,:],color = 'indigo')
        axs[0].set_ylabel("direction cable 1")    
        axs[0].axhline(y = 0, color = 'r', linestyle = '--')    

        axs[1].plot(t_x,direct_Log[1,:],color = 'indigo')
        axs[1].set_ylabel("direction cable 2")    
        axs[1].axhline(y = 0, color = 'r', linestyle = '--')   

        for i in range(2):
            axs[i].set_xlim([0,self.t_sim])
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
        fig.tight_layout()
        # plt.show()
    
    def plot_error(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim+1)
        err_x_Log = self.data['err_x'] 
        err_the_Log = self.data['err_the']
        trigger_Log = self.data['trigger']

        # -- position
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(3,1)
        # fig.suptitle("Tracking",fontsize = 22)

        axs[0].plot(t_x,err_x_Log[0,:],color = 'red', linewidth=2)
        axs[0].set_ylabel("error x")

        axs[1].plot(t_x,err_x_Log[1,:],color = 'green', linewidth=2)
        axs[1].set_ylabel("error y")

        axs[2].plot(t_x,err_the_Log[0,:],color = 'blue', linewidth=2)
        axs[2].set_ylabel("error theta [rad]")

        for i in range(3):
                axs[i].set_xlim([0,self.t_sim])
                axs[i].grid()
                axs[i].ticklabel_format(useOffset=False)
                for trig in range(trigger_Log.shape[0]):
                    axs[i].axvline(x=trigger_Log[trig], color='indigo', linestyle='--', linewidth=1)
        fig.tight_layout()

    def plot_error_all(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim+1)
        err_x_Log = self.data['err_x'] 
        err_v_Log = self.data['err_v'] 
        err_the_Log = self.data['err_the']
        err_thedot_Log = self.data['err_thedot']
        trigger_Log = self.data['trigger']

        # -- position
        plt.rcParams['figure.figsize'] = [6, 10]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(6,1)
        # fig.suptitle("Tracking",fontsize = 22)

        axs[0].plot(t_x,err_x_Log[0,:],color = 'red', linewidth=2)
        axs[0].set_ylabel("error x [m]")

        axs[1].plot(t_x,err_x_Log[1,:],color = 'green', linewidth=2)
        axs[1].set_ylabel("error y [m]")

        axs[2].plot(t_x,err_the_Log[0,:],color = 'blue', linewidth=2)
        axs[2].set_ylabel("error theta [rad]")

        axs[3].plot(t_x,err_v_Log[0,:],color = 'red', linewidth=2)
        axs[3].set_ylabel("error xdot [m/s]")

        axs[4].plot(t_x,err_v_Log[1,:],color = 'green', linewidth=2)
        axs[4].set_ylabel("error ydot [m/s]")

        axs[5].plot(t_x,err_thedot_Log[0,:],color = 'blue', linewidth=2)
        axs[5].set_ylabel("error thetadot [rad/s]")

        for i in range(6):
                axs[i].set_xlim([0,self.t_sim])
                axs[i].grid()
                axs[i].ticklabel_format(useOffset=False)
                for trig in range(trigger_Log.shape[0]):
                    axs[i].axvline(x=trigger_Log[trig], color='indigo', linestyle='--', linewidth=1)
        fig.tight_layout()
    
    def plot_states_track(self,name):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim+1)
        x_Log = self.data['x'] 
        xref_Log = self.data['xref']
        trigger_Log = self.data['trigger']
        
        if name == 'transporter1':
            num = 0
        elif name == 'transporter2':
            num = 1
        elif name == 'load':
            num = 2

        # -- position
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(3,1)
        fig.suptitle(name,fontsize = 16)

        axs[0].plot(t_x,x_Log[0 + num*6,:],label='px',color = 'red')
        axs[0].set_ylabel("px [m]")
        axs[0].step(t_x,xref_Log[0+num*6,:], linewidth=2,color = 'darkred', linestyle='dashed', where='post',label='px_target')
        
        axs[1].plot(t_x,x_Log[1 + num*6,:],label='py',color = 'green')
        axs[1].set_ylabel("py [m]")
        axs[1].step(t_x,xref_Log[1+num*6,:], linewidth=2,color = 'darkgreen', linestyle='dashed', where='post',label='py_target')
        
        axs[2].plot(t_x,x_Log[4 + num*6,:],label='theta',color = 'blue')
        axs[2].set_ylabel("theta [rad]")
        axs[2].step(t_x,xref_Log[4+num*6,:], linewidth=2,color = 'darkblue', linestyle='dashed', where='post',label='theta_target')
        axs[2].set_xlabel("Time [s]")

        for i in range(3):
            axs[i].set_xlim([0,self.t_sim])
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
            axs[i].legend()
            for trig in range(trigger_Log.shape[0]):
                axs[i].axvline(x=trigger_Log[trig], color='black', linestyle='--', linewidth=1)
        fig.tight_layout()
        # plt.show()
    
    def plot_input(self,name):
        if name == 'transporter1':
            num = 0
        elif name == 'transporter2':
            num = 1
        ulim = self.data['ulim']
        plt.rcParams['figure.figsize'] = [6, 8]
        plt.rcParams['figure.dpi'] = 100
        u_Log = self.data['u']
        t_u = np.linspace(0,self.t_sim,self.n_sim)
        fig, axs = plt.subplots(4,1)
        fig.suptitle(name,fontsize = 10)
        for i in range(4):
                axs[i].set_xlim([0,self.t_sim])
                axs[i].grid()
                axs[i].axhline(y=ulim, color='k', linestyle='--', linewidth=2)
                axs[i].axhline(y=-ulim, color='k', linestyle='--', linewidth=2)
                axs[i].ticklabel_format(useOffset=False)

        axs[0].step(t_u,u_Log[0+num*4,:],label='u1', where='post', linewidth=2)
        axs[0].set_ylabel("u1 [N]")

        axs[1].step(t_u,u_Log[1+num*4,:],label='u2', where='post', linewidth=2)
        axs[1].set_ylabel("u2 [N]")

        axs[2].step(t_u,u_Log[2+num*4,:],label='u3', where='post', linewidth=2)
        axs[2].set_ylabel("u3 [N]")

        axs[3].step(t_u,u_Log[3+num*4,:],label='u4', where='post', linewidth=2)
        axs[3].set_ylabel("u4 [Nm]")
        axs[3].set_xlabel("Time [s]")

        
        fig.tight_layout()
        # plt.show()

    def plot_input_vec(self,name):
        if name == 'transporter1':
            num = 0
            D = self.model.D1
            L = self.model.L1
        elif name == 'transporter2':
            num = 1
            D = self.model.D2
            L = self.model.L2
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        u_Log = self.data['u']
        t_u = np.linspace(0,self.t_sim,self.n_sim)
        fig, axs = plt.subplots(3,1)
        fig.suptitle(name,fontsize = 10)
        F = D @ u_Log[num*4:num*4+4,:]
        T = L @ u_Log[num*4:num*4+4,:]

        axs[0].step(t_u,np.reshape(F[0,:],(-1,1)),label='Fx', where='post')
        axs[0].set_ylabel("Fx [N]")

        axs[1].step(t_u,np.reshape(F[1,:],(-1,1)),label='Fy', where='post')
        axs[1].set_ylabel("Fy [N]")

        axs[2].step(t_u,np.reshape(T[0,:],(-1,1)),label='Tz', where='post')
        axs[2].set_ylabel("Tz [Nm]")
        axs[2].set_xlabel("Time [s]")

        for i in range(3):
            axs[i].set_xlim([0,self.t_sim])
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
        fig.tight_layout()
        # plt.show()
    
    def plot_traj(self):
        x_Log = self.data['x']
        xref_Log = self.data['xref']
        fig, ax = plt.subplots(1,1)
        ax.plot(xref_Log[12,:],xref_Log[13,:], color = 'k', linestyle = '--',linewidth = 1,label='desired path', marker = 'x')
        ax.plot(x_Log[12,:],x_Log[13,:], linestyle='-',linewidth = 2,color = 'seagreen',label='actual path')
        # ax.set_xlim([-2.5,2.5])
        # ax.set_ylim([-1.5,1.5])
        ax.grid()
        ax.set_ylabel("y [m]")
        ax.set_xlabel("x [m]")
        ax.set_aspect('equal')
        ax.legend()
    
    def plot_traj_3D(self):
        x_Log = self.data['x']
        xref_Log = self.data['xref']
        fig = plt.figure(figsize=(10,10))
        axs = fig.add_subplot(projection='3d')
        axs.view_init(elev=90, azim=-90)
        axs.set_xlim([-2.5,2.5])
        axs.set_ylim([-1.5,1.5])
        axs.set_zlim([-1,1])
        axs.grid()
        axs.plot3D(xref_Log[12,:],xref_Log[13,:],np.zeros_like(xref_Log[12,:]), color = 'k', linestyle = '--',linewidth = 1)
        axs.plot3D(x_Log[12,:],x_Log[13,:],np.zeros_like(x_Log[12,:]), linestyle='-',linewidth = 2,color = 'seagreen')

