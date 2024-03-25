'''
Create environment for trajectory tracking task
'''
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transporter.utils.function import *
from transporter.simulators.visualizer import *


class Simulator2D:
    def __init__(self,model,controller):        
        self.dt = model.dt
        self.n = model.n
        self.m = model.m
        self.model = model
        self.controller = controller
        print('-----------------Simulator initialized-----------------')
    
    def run(self,init,simulation_time,DEBUG = False, verbose = True,SLEEP = True):
        self.t_sim = simulation_time
        self.n_sim = int(self.t_sim/self.dt)
        x0 = init['x0']
        x_warm = init['x_warm']
        u_warm = init['u_warm']
        xref_array = init['xref']  
        if SLEEP:
            sleep_time = 5
        else:
            sleep_time = 0
        sleep_time_count = 0     

        # data logger
        data = {}
        x_Log = np.zeros((self.n, self.n_sim+1))
        x_pred_Log = np.zeros((self.n_sim+1, self.n, self.controller.Nt+1))
        u_Log = np.zeros((self.m, self.n_sim))
        cost_Log = np.zeros((1,self.n_sim))
        compute_time_Log = np.zeros((1,self.n_sim))
        xref_Log = np.zeros((self.n, self.n_sim+1))
        trigger_Log = np.zeros((xref_array.shape[1]))

        # error load
        err_x_Log = np.zeros((2, self.n_sim+1))
        err_the_Log = np.zeros((1, self.n_sim+1))

        # initialized values
        x_Log[:,[0]] = x0
        point = 0
        xref = xref_array[:,[point]]
        xref_Log[:,[0]] = xref
        err_x_Log[:,[0]] = xref[12:14] - x0[12:14]
        err_the_Log[:,[0]] = xref[16] - x0[16]

      
        if DEBUG:
            # cable length overtime
            l_Log = np.zeros((2, self.n_sim+1))
            l_Log[:,[0]] = ca.DM([[self.model.l1],[self.model.l2]])

            # n1, n2 direction overtime
            direct_Log = np.zeros((2,self.n_sim))

        for i in range(self.n_sim):     
            if verbose:    
                print(">> t_sim: " + str((i+1)*self.dt)+" s")
            t_start = time.perf_counter()
            u_Log[:,[i]],x_temp,u_temp = self.controller.solve_mpc(x_Log[:,i],xref,x_warm,u_warm)
            # u_Log[:,[i]] = (np.array([0.1,0.1,0.5,0.5,0.1,0.1,-0.5,-0.5])*1.4).reshape(-1,1)
  

            t_stop = time.perf_counter()
            time_elapse = t_stop-t_start
            compute_time_Log[:,[i]] = time_elapse

            n1_temp = self.model.n1F(x_Log[:,[i]], u_Log[:,[i]])
            n2_temp = self.model.n2F(x_Log[:,[i]], u_Log[:,[i]])
            Tv1_temp = self.model.tension1(x_Log[:,[i]], u_Log[:,[i]])
            Tv2_temp = self.model.tension2(x_Log[:,[i]], u_Log[:,[i]])

            if verbose:
                print("Compute time: "+ str(time_elapse))  

            # -- Log prediction states in each time step      
            x_pred_Log[i,:,:] = x_temp       

            # -- create initial guess states for solver by remove the initial element 
            # and copy the last element 
            x_warm = np.hstack((x_temp[:,1:],np.reshape(x_temp[:,-1],(-1,1))))
            u_warm = u_temp[:,0:]


            # -- use smaller dt for simulation
            x_sim_temp = x_Log[:,[i]]
            for sim_step in range(self.model.scale):
                x_sim_temp = self.model.rk4sim(x_sim_temp, u_Log[:,[i]])
            x_Log[:,[i+1]] = x_sim_temp
            cost_Log[:,[i]] = self.controller.running_cost(x_Log[:,[i]],xref,self.controller.Q,u_Log[:,[i]],self.controller.R)
            err_x_Log[:,[i+1]] = xref[12:14] - x_Log[12:14,[i+1]]
            err_the_Log[:,[i+1]] = xref[16] - x_Log[16,[i+1]]

            if verbose:
                print("MPC cost: " + str(cost_Log[:,[i]][0][0]))
                print("MPC terminal cost: " + str(self.controller.terminal_cost(x_Log[:,[i]],xref,self.controller.P)))
                # print("Error position [m]: " + str(np.linalg.norm(err_x_Log[:,[i+1]])))
                # print("Error angle [rad]: " + str(err_the_Log[:,[i+1]][0][0]))
            
            if SLEEP:
                position_trigger_error = 0.005
                angle_trigger_error = 0.01
            else:
                position_trigger_error = 0.01#0.03
                angle_trigger_error = 0.1
            

            
            if np.linalg.norm(err_x_Log[:,[i+1]]) <= position_trigger_error and abs(err_the_Log[:,[i+1]])<=angle_trigger_error:
                sleep_time_count = sleep_time_count + self.dt
                if verbose:
                    print('################ point'+str(point+1)+' ################')
                if sleep_time_count >= sleep_time:
                    sleep_time_count = 0
                    if point == xref_array.shape[1]-1:
                        print('######################## FINAL POINT ########################')
                        print(">>>>>>>>>>>>>>> Final t_sim: " + str((i+1)*self.dt)+" s")
                    else:
                        point = point + 1  
                        xref = xref_array[:,[point]]      
                        trigger_Log[point] = (i+1)*self.dt   
            else:
                sleep_time_count = 0   
            xref_Log[:,[i+1]] = xref

            if DEBUG:
                # -- check that cables are always in tension or not
                direct1_temp = n1_temp.T @ Tv1_temp #/ (Tv1_temp.T @ Tv1_temp)
                direct2_temp = n2_temp.T @ Tv2_temp #/ (Tv2_temp.T @ Tv2_temp)
                direct_Log[:,[i]] = np.vstack([direct1_temp,direct2_temp])
       
        data['x'] = x_Log
        data['u'] = u_Log
        data['xref_array'] = xref_array
        data['xref'] = xref_Log
        data['err_x'] = err_x_Log
        data['err_the'] = err_the_Log
        data['cost'] = cost_Log
        data['compute_time'] = compute_time_Log
        data['x_pred'] = x_pred_Log
        data['t_sim'] = self.t_sim
        data['n_sim'] = self.n_sim
        data['trigger'] = trigger_Log
        data['ulim'] = self.controller.ulim
        
        if DEBUG:
            data['direct'] = direct_Log
        self.data = data
    
    def plot_states(self,name):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim+1)
        x_Log = self.data['x'] 
        q_Log = self.data['q']
        if name == 'transporter1':
            num = 0
        elif name == 'transporter2':
            num = 1
        elif name == 'load':
            num = 2

        # -- position
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(3,2)
        fig.suptitle(name,fontsize = 22)

        axs[0,0].plot(t_x,x_Log[0 + num*13,:],label='px')
        axs[0,0].set_ylabel("px [m]")
        
        axs[1,0].plot(t_x,x_Log[1 + num*13,:],label='py')
        axs[1,0].set_ylabel("py [N]")

        axs[2,0].plot(t_x,x_Log[2 + num*13,:],label='pz')
        axs[2,0].set_ylabel("pz [N]")
        axs[2,0].set_xlabel("Time [s]")

        axs[0,1].plot(t_x,x_Log[3 + num*13,:],label='vx')
        axs[0,1].set_ylabel("vx [m/s]")

        axs[1,1].plot(t_x,x_Log[4 + num*13,:],label='vy')
        axs[1,1].set_ylabel("vy [m/s]")

        axs[2,1].plot(t_x,x_Log[5 + num*13,:],label='vz')
        axs[2,1].set_ylabel("vz [m/s]")
        axs[2,1].set_xlabel("Time [s]")

        for i in range(3):
            for j in range(2):
                axs[i,j].set_xlim([0,self.t_sim])
                axs[i,j].grid()
                axs[i,j].ticklabel_format(useOffset=False)
        fig.tight_layout()

        # -- attitude
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(4,2)
        fig.suptitle(name,fontsize = 22)

        axs[0,0].plot(t_x,x_Log[6 + num*13,:],label='q1')
        axs[0,0].set_ylabel("q1")

        axs[1,0].plot(t_x,x_Log[7 + num*13,:],label='q2')
        axs[1,0].set_ylabel("q2")

        axs[2,0].plot(t_x,x_Log[8 + num*13,:],label='q3')
        axs[2,0].set_ylabel("q3")

        axs[3,0].plot(t_x,x_Log[9 + num*13,:],label='q4')
        axs[3,0].set_ylabel("q4")
        axs[3,0].set_xlabel("Time [s]")

        axs[0,1].plot(t_x,x_Log[10 + num*13,:],label='wx')
        axs[0,1].set_ylabel("wx [rad/s]")

        axs[1,1].plot(t_x,x_Log[11 + num*13,:],label='wy')
        axs[1,1].set_ylabel("wy [rad/s]")

        axs[2,1].plot(t_x,x_Log[12 + num*13,:],label='wz')
        axs[2,1].set_ylabel("wz [rad/s]")

        axs[3,1].plot(t_x,q_Log[num,:],label='q')
        axs[3,1].set_ylabel("norm q")
        # axs[3,1].set_ylim([0.999,1.001])
        axs[3,1].ticklabel_format(useOffset=False)

        axs[3,1].set_xlabel("Time [s]")
        for i in range(4):
            axs[i,0].set_ylim([-1.1,1.1])
        for i in range(4):
            for j in range(2):
                axs[i,j].set_xlim([0,self.t_sim])
                axs[i,j].grid()
                axs[i,j].ticklabel_format(useOffset=False)
        fig.tight_layout()
        plt.show()
    
    def plot_input(self,name):
        if name == 'transporter1':
            num = 0
        elif name == 'transporter2':
            num = 1
        ulim = 10
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        u_Log = self.data['u']
        t_u = np.linspace(0,self.t_sim,self.n_sim)
        fig, axs = plt.subplots(3,2)
        fig.suptitle(name,fontsize = 22)
        axs[0,0].step(t_u,u_Log[0+num*6,:],label='u1', where='post')
        axs[0,0].set_ylabel("u1 [N]")

        axs[1,0].step(t_u,u_Log[1+num*6,:],label='u2', where='post')
        axs[1,0].set_ylabel("u2 [N]")

        axs[2,0].step(t_u,u_Log[2+num*6,:],label='u3', where='post')
        axs[2,0].set_ylabel("u3 [N]")
        axs[2,0].set_xlabel("Time [s]")

        axs[0,1].step(t_u,u_Log[3+num*6,:],label='u4', where='post')
        axs[0,1].set_ylabel("u4 [Nm]")

        axs[1,1].step(t_u,u_Log[4+num*6,:],label='u5', where='post')
        axs[1,1].set_ylabel("u5 [Nm]")

        axs[2,1].step(t_u,u_Log[5+num*6,:],label='u6', where='post')
        axs[2,1].set_ylabel("u6 [Nm]")
        axs[2,1].set_xlabel("Time [s]")

        for i in range(3):
            for j in range(2):
                axs[i,j].set_xlim([0,self.t_sim])
                axs[i,j].grid()
                axs[i,j].axhline(y=ulim, color='k', linestyle='--', linewidth=2)
                axs[i,j].axhline(y=-ulim, color='k', linestyle='--', linewidth=2)
                axs[i,j].ticklabel_format(useOffset=False)
        fig.tight_layout()
        plt.show()

    def plot_input_vec(self,name):
        if name == 'transporter1':
            num = 0
            D = self.model.D1
            L = self.model.L1
        elif name == 'transporter2':
            num = 1
            D = self.model.D2
            L = self.model.L2
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        u_Log = self.data['u']
        t_u = np.linspace(0,self.t_sim,self.n_sim)
        fig, axs = plt.subplots(3,2)
        fig.suptitle(name,fontsize = 22)
        F = D @ u_Log[num*6:num*6+6,:]
        T = L @ u_Log[num*6:num*6+6,:]
        for i in range(3):
            for j in range(2):
                axs[i,j].set_xlim([0,self.t_sim])
                axs[i,j].grid()
                axs[i,j].ticklabel_format(useOffset=False)

        axs[0,0].step(t_u,np.reshape(F[0,:],(-1,1)),label='Fx', where='post')
        axs[0,0].set_ylabel("Fx [N]")

        axs[1,0].step(t_u,np.reshape(F[1,:],(-1,1)),label='Fy', where='post')
        axs[1,0].set_ylabel("Fy [N]")

        axs[2,0].step(t_u,np.reshape(F[2,:],(-1,1)),label='Fz', where='post')
        axs[2,0].set_ylabel("Fz [N]")
        axs[2,0].set_xlabel("Time [s]")

        axs[0,1].step(t_u,np.reshape(T[0,:],(-1,1)),label='Tx', where='post')
        axs[0,1].set_ylabel("Tx [Nm]")

        axs[1,1].step(t_u,np.reshape(T[1,:],(-1,1)),label='Ty', where='post')
        axs[1,1].set_ylabel("Ty [Nm]")

        axs[2,1].step(t_u,np.reshape(T[2,:],(-1,1)),label='Tz', where='post')
        axs[2,1].set_ylabel("Tz [Nm]")
        axs[2,1].set_xlabel("Time [s]")


        fig.tight_layout()
        plt.show()
   
    def plot_tension(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim)
        direct_Log = self.data['direct'] 

        plt.rcParams['figure.figsize'] = [6, 3]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(2,1)
        # fig.suptitle("Cable Tension",fontsize = 22)

        axs[0].plot(t_x,direct_Log[0,:],color = 'darkorange')
        axs[0].set_ylabel("direction cable 1")        

        axs[1].plot(t_x,direct_Log[1,:],color = 'darkorange')
        axs[1].set_ylabel("direction cable 2")    

        for i in range(2):
            axs[i].set_xlim([0,self.t_sim])
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
        fig.tight_layout()
        plt.show()
   
    def plot_cost(self):
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

        axs[1].plot(t_cost[1:],time_Log[0,1:],label='time',marker=".") # skip the initialized time for better scale
        axs[1].set_ylabel("Computed Time [s]")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid()
        axs[1].set_xlim([0,self.t_sim])
        axs[1].axhline(y = np.mean(time_Log[0,1:]), color = 'r', linestyle = '--')
        print("Max initialized computed time: " + str(np.max(time_Log))+" s")
        print("Max computed time: " + str(np.max(time_Log[0,1:]))+" s")
        print("Average computed time: "+ str(np.mean(time_Log[0,1:]))+" s")
        fig.tight_layout()
        plt.show()
    
    def plot_track(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim+1)
        x_Log = self.data['x'] 
        xref_Log = self.data['xref']
        err_x_Log = self.data['err_x'] 
        err_the_Log = self.data['err_the']

        # -- position
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(3,2)
        fig.suptitle("Tracking",fontsize = 22)

        axs[0,0].plot(t_x,x_Log[12,:],label='px')
        axs[0,0].set_ylabel("px [m]")        
        axs[0,0].step(t_x,xref_Log[12,:], linewidth=2,color = 'black', linestyle='dashed', where='post')
                
        axs[1,0].plot(t_x,x_Log[13,:],label='py')
        axs[1,0].set_ylabel("py [m]")
        axs[1,0].step(t_x,xref_Log[13,:], linewidth=2,color = 'black', linestyle='dashed', where='post')

        axs[2,0].plot(t_x,x_Log[16,:],label='theta')
        axs[2,0].set_ylabel("theta [rad]")
        axs[2,0].step(t_x,xref_Log[16,:], linewidth=2,color = 'black', linestyle='dashed', where='post')

        axs[0,1].plot(t_x,np.reshape(np.square(err_x_Log[0:2,:]).sum(axis=0),(-1,1)))
        axs[0,1].set_ylabel("norm error position")

        axs[1,1].plot(t_x,np.reshape(err_the_Log,(-1,1)))
        axs[1,1].set_ylabel("norm error angle")

        for i in range(3):
            for j in range(2):
                axs[i,j].set_xlim([0,self.t_sim])
                axs[i,j].grid()
                axs[i,j].ticklabel_format(useOffset=False)
        fig.tight_layout()

    def animate(self,save = False,name = '',fps=10,bitrate=1000,dpi = 1000, PREDICT = False):
        x_Log = self.data['x'] 
        u_Log = self.data['u']
        x_pred_Log = self.data['x_pred']
        xref_array = self.data['xref_array']

        def toggle_animation(event):
            if event.key == 'p':
                animation.Animation.pause(self.anim)
            elif event.key == 'r':
                animation.Animation.resume(self.anim)


        def update_frame(i):
            # == Extract variable for visualization    
            # -- states robot 1
            p1_vis = np.hstack([x_Log[0:2,i+1],0])
            v1_vis = np.hstack([x_Log[2:4,i+1],0]).reshape((-1, 1))
            the1_vis = x_Log[4,i+1]
            q1_vis = angle2quatZ(the1_vis)
            w1_vis = np.hstack([0,0,x_Log[5,i+1]])            

            # -- states robot 2
            p2_vis = np.hstack([x_Log[6:8,i+1],0])
            v2_vis = np.hstack([x_Log[8:10,i+1],0]).reshape((-1, 1))
            the2_vis = x_Log[10,i+1]
            q2_vis = angle2quatZ(the2_vis)
            w2_vis = np.hstack([0,0,x_Log[11,i+1]]) 

            # -- states Load
            pL_vis = np.hstack([x_Log[12:14,i+1],0])
            vL_vis = np.hstack([x_Log[14:16,i+1],0]).reshape((-1, 1))
            theL_vis = x_Log[16,i+1]
            qL_vis = angle2quatZ(theL_vis)
            wL_vis = np.hstack([0,0,x_Log[17,i+1]]) 

            # -- states connection
            p1_prime_vis = np.array(p1_vis + dcm_e2b(q1_vis).T @ ca.vertcat(self.model.r1,0))
            s1_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ ca.vertcat(self.model.R1,0))
            p2_prime_vis = np.array(p2_vis + dcm_e2b(q2_vis).T @ ca.vertcat(self.model.r2,0))
            s2_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ ca.vertcat(self.model.R2,0))

            # -- inputs --
            F1_vis = np.array(dcm_e2b(q1_vis).T @ np.vstack([self.model.D1 @ u_Log[0:4,i],0]))          
            T1_vis = dcm_e2b(q1_vis).T @ np.vstack([0,0,self.model.L1 @ u_Log[0:4,i]])                  
            F2_vis = np.array(dcm_e2b(q2_vis).T @ np.vstack([self.model.D2 @ u_Log[4:8,i],0]))                 
            T2_vis = dcm_e2b(q2_vis).T @ np.vstack([0,0,self.model.L2 @ u_Log[4:8,i]]) 
        
            # -- cable tension
            Tv1_vis = ca.vertcat(self.model.tension1(xk = x_Log[:,i+1], uk = u_Log[:,i])['Tv1'],0)
            Tv2_vis = ca.vertcat(self.model.tension2(xk = x_Log[:,i+1], uk = u_Log[:,i])['Tv2'],0)

            # -- cable length
            l1 = np.sqrt((p1_prime_vis - s1_vis).T @ (p1_prime_vis - s1_vis))
            l2 = np.sqrt((p2_prime_vis - s2_vis).T @ (p2_prime_vis - s2_vis))

            # -- visualization
            ax.clear()
            plot_robot(p1_vis,q1_vis,ax,clr='red',cle='darkred')
            plot_robot(p2_vis,q2_vis,ax,clr='blue',cle='darkblue')
            plot_load(pL_vis,qL_vis,ax)
            plot_cable(p1_prime_vis,s1_vis,ax)
            plot_cable(p2_prime_vis,s2_vis,ax)
            # -- force_torque
            plot_vector(p1_vis,F1_vis,ax,scale=0.1)
            plot_vector(p2_vis,F2_vis,ax,scale=0.1)
            plot_vector(p1_vis,T1_vis,ax,clr='c',scale=0.1)
            plot_vector(p2_vis,T2_vis,ax,clr='c',scale=0.1)
            # -- tension
            plot_vector(p1_prime_vis,-Tv1_vis,ax,clr='lime',scale=0.1)
            plot_vector(s1_vis,Tv1_vis,ax,clr='lime',scale=0.1)
            plot_vector(p2_prime_vis,-Tv2_vis,ax,clr='lime',scale=0.1)
            plot_vector(s2_vis,Tv2_vis,ax,clr='lime',scale=0.1)
            # -- velocity
            plot_vector(p1_vis,v1_vis,ax,clr='orange',scale=0.3)
            plot_vector(p2_vis,v2_vis,ax,clr='orange',scale=0.3)
            plot_vector(pL_vis,vL_vis,ax,clr='orange',scale=0.3)

            # -- reference axis
            # plot_axis(-1,-1.5,0,np.array([0,0,0,1]),ax,size=1)

            # -- trajectory
            ax.plot3D(x_Log[12,:],x_Log[13,:],np.zeros_like(x_Log[12,:]), linestyle='dashed')


            for point in range(np.shape(xref_array)[1]):
                xref = xref_array[:,[point]]
                plot_path(xref,ax, alpha = 0.7,PLANAR=True)

            # -- plot prediction ghost
            if PREDICT:
                # plot_system(ax,i,x_pred_Log[i,:,int(self.controller.Nt/6):int(self.controller.Nt/6)+1],self.model, alpha = 0.2,PLANAR=True)
                # plot_system(ax,i,x_pred_Log[i,:,int(self.controller.Nt/4):int(self.controller.Nt/4)+1],self.model, alpha = 0.2,PLANAR=True)
                plot_system(ax,i,x_pred_Log[i,:,int(self.controller.Nt/2):int(self.controller.Nt/2)+1],self.model, alpha = 0.2,PLANAR=True)
                plot_system(ax,i,x_pred_Log[i,:,self.controller.Nt:],self.model, alpha = 0.5,PLANAR=True)

            # ax.plot3D([p1_vis[0], p1_vis[0] + F1_vis[0]],[p1_vis[1], p1_vis[1] + F1_vis[1]],[p1_vis[2], p1_vis[2] + F1_vis[2]])
            ax.set_proj_type('ortho') 
            # ax.set_xlim([-1.5,0.5])
            # ax.set_ylim([-1.5,1.5])
            # ax.set_zlim([-1,1])
            ax.set_xlim([-2.5,2.5])
            ax.set_ylim([-1.5,1.5])
            ax.set_zlim([-1,1])
            ax.set_aspect('equal')
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.set_title("$\\textbf{Centralized Control}$",fontsize=20)
            ax.grid()    
            time_text = ax.text2D(0.05, 0.9, time_template, transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9),
                ha='left', va='top')
            input_text = ax.text2D(0.75, 0.15, input_template, transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9),
                ha='left', va='top')
            state_text = ax.text2D(-0.1, 0.15,state_template, transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9),
                ha='left', va='top')
            time_text.set_text(time_template % ((i+1)*self.dt,
                                                l1,
                                                l2,
                                                np.linalg.norm(Tv1_vis),
                                                np.linalg.norm(Tv2_vis)))
            input_text.set_text(input_template % (F1_vis[0],F2_vis[0],
                                                F1_vis[1],F2_vis[1],
                                                F1_vis[2],F2_vis[2],
                                                T1_vis[0],T2_vis[0],
                                                T1_vis[1],T2_vis[1],
                                                T1_vis[2],T2_vis[2]))
            state_text.set_text(state_template % (str(trunc(p1_vis, decs=2)),str(trunc(p2_vis, decs=2)),str(trunc(pL_vis, decs=2)),
                                                str(trunc(*v1_vis.reshape((1, -1)), decs=2)),str(trunc(*v2_vis.reshape((1, -1)), decs=2)),str(trunc(*vL_vis.reshape((1, -1)), decs=2)),
                                                str(trunc(the1_vis*180/np.pi, decs=2)),str(trunc(the2_vis*180/np.pi, decs=2)),str(trunc(theL_vis*180/np.pi, decs=2)),
                                                str(trunc(w1_vis, decs=2)),str(trunc(w2_vis, decs=2)),str(trunc(wL_vis, decs=2))))
            return ax

        fig = plt.figure(figsize=(10,10))
        plt.rcParams['text.usetex'] = True
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=90, azim=-90)
        time_template = """time = %.2f s cable1: %.10f m cable2: %.10f m Ts1: %.5f N Ts2: %.5f N"""
        input_template = """F1x: %.3f N   F2x: %.3f N  
                        \nF1y: %.3f N   F2y: %.3f N
                        \nF1z: %.3f N   F2z: %.3f N
                        \nT1x: %0.3f Nm  T2x: %0.3f Nm
                        \nT1y: %0.3f Nm  T2y: %0.3f Nm
                        \nT1z: %0.3f Nm  T2z: %0.3f Nm"""
        state_template = """p1: %s m     p2: %s m     pL: %s m 
                        \nv1: %s m/s     v2: %s m/s     vL: %s m/s
                        \nthe1: %s deg   the2: %s deg   theL: %s deg
                        \nw1: %s rad/s  w2: %s rad/s   wL: %s rad/s"""

        # Create the animation
        fig.canvas.mpl_connect('key_press_event', toggle_animation)
        self.anim = animation.FuncAnimation(fig, update_frame, frames = np.arange(0, self.n_sim), interval=10) # interval = Delay between frames in milliseconds.
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=bitrate)
            self.anim.save('' + name + '.mp4', writer=writer,dpi=dpi)
        # anim.save('animation.gif')
        plt.show()
    



