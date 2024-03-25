'''
Create class for multi-agent load transportation system
'''
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transporter.utils.function import *
from transporter.simulators.visualizer import *

class Simulator3A:
    def __init__(self, model,controller):        
        self.dt = model.dt
        self.n = model.n
        self.m = model.m
        self.model = model
        self.controller = controller
        print('-----------------Simulator initialized-----------------')
    
    def run(self,init,simulation_time,DEBUG = False, verbose = True):
        print('-----------------Start Simulation-----------------')
        self.t_sim = simulation_time
        self.n_sim = int(self.t_sim/self.dt)
        x0 = init['x0']
        x_warm = init['x_warm']
        u_warm = init['u_warm']
        xref_array = init['xref']       

        # -- data logger
        data = {}
        x_Log = np.zeros((self.n, self.n_sim+1))
        x_pred_Log = np.zeros((self.n_sim+1, self.n, self.controller.Nt+1))
        u_Log = np.zeros((self.m, self.n_sim))
        q_Log = np.zeros((3,self.n_sim+1))
        # deg_Log = np.zeros((9,self.n_sim+1))
        cost_Log = np.zeros((1,self.n_sim))
        compute_time_Log = np.zeros((1,self.n_sim))
        xref_Log = np.zeros((self.n, self.n_sim+1))
        trigger_Log = np.zeros((xref_array.shape[1]))

        # -- error load
        err_x_Log = np.zeros((3, self.n_sim+1))
        err_q_Log = np.zeros((1,self.n_sim+1))

        # -- initialized values
        x_Log[:,[0]] = x0
        point = 0
        xref = xref_array[:,[point]]
        xref_Log[:,[0]] = xref
        err_x_Log[:,[0]] = xref[39:42] - x0[39:42]
        err_q_Log[:,[0]] = quatdist(xref[45:49],x_Log[45:49,[0]])
      
        if DEBUG:
            # cable length overtime
            l_Log = np.zeros((3, self.n_sim+1))
            l_Log[:,[0]] = ca.DM([[self.model.l1],[self.model.l2],[self.model.l3]])

            # n1, n2 direction overtime
            direct_Log = np.zeros((3,self.n_sim))

        # quaternion size overtime
        q_Log[0,0] = ca.norm_2(x0[6:10,0])
        q_Log[1,0] = ca.norm_2(x0[19:23,0])
        q_Log[2,0] = ca.norm_2(x0[32:36,0])

        for i in range(self.n_sim):         
            print(">> t_sim: " + str((i+1)*self.dt)+" s")
            t_start = time.perf_counter()
            u_Log[:,[i]],x_temp,u_temp = self.controller.solve_mpc(x_Log[:,i],xref,x_warm,u_warm)
            t_stop = time.perf_counter()
            time_elapse = t_stop-t_start
            compute_time_Log[:,[i]] = time_elapse

            # -- compute helper function for cable
            n1_temp = self.model.n1F(x_Log[:,[i]], u_Log[:,[i]])
            n2_temp = self.model.n2F(x_Log[:,[i]], u_Log[:,[i]])
            n3_temp = self.model.n3F(x_Log[:,[i]], u_Log[:,[i]])
            Tv1_temp = self.model.tension1(x_Log[:,[i]], u_Log[:,[i]])
            Tv2_temp = self.model.tension2(x_Log[:,[i]], u_Log[:,[i]])
            Tv3_temp = self.model.tension3(x_Log[:,[i]], u_Log[:,[i]])

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
            q_Log[0,[i+1]] = ca.norm_2(x_Log[:,[i+1]][6:10])
            q_Log[1,[i+1]] = ca.norm_2(x_Log[:,[i+1]][19:23])
            q_Log[2,[i+1]] = ca.norm_2(x_Log[:,[i+1]][32:36])

            cost_Log[:,[i]] = self.controller.running_cost(x_Log[:,[i]],xref,self.controller.Q,u_Log[:,[i]],self.controller.R)
            err_x_Log[:,[i+1]] = xref[39:42] - x_Log[39:42,[i+1]]
            err_q_Log[:,[i+1]] = quatdist(xref[45:49],x_Log[45:49,[i+1]])
            if verbose:
                print("MPC cost: " + str(cost_Log[:,[i]][0][0]))
            
            if np.square(err_x_Log[:,[i+1]]).sum(axis=0) <= 0.1 and err_q_Log[:,[i+1]]<=0.05:
                print('################ point'+str(point+1)+' ################')
                if point == xref_array.shape[1]-1:
                    print('################ FINAL POINT ################')
                else:
                    point = point + 1  
                    xref = xref_array[:,[point]]  
                    trigger_Log[point] = (i+1)*self.dt       
            xref_Log[:,[i+1]] = xref

            if DEBUG:
                # -- check that cables are always in tension or not
                direct1_temp = n1_temp.T @ Tv1_temp #/ (Tv1_temp.T @ Tv1_temp)
                direct2_temp = n2_temp.T @ Tv2_temp #/ (Tv2_temp.T @ Tv2_temp)
                direct3_temp = n3_temp.T @ Tv3_temp #/ (Tv3_temp.T @ Tv3_temp)
                direct_Log[:,[i]] = np.vstack([direct1_temp,direct2_temp,direct3_temp])

        
        data['x'] = x_Log
        data['u'] = u_Log
        data['q'] = q_Log
        data['xref_array'] = xref_array
        data['xref'] = xref_Log
        data['err_x'] = err_x_Log
        data['err_q'] = err_q_Log
        data['cost'] = cost_Log
        data['compute_time'] = compute_time_Log
        data['x_pred'] = x_pred_Log
        data['trigger'] = trigger_Log
        data['t_sim'] = self.t_sim
        data['n_sim'] = self.n_sim
        data['ulim'] = self.controller.ulim

        if DEBUG:
            data['direct'] = direct_Log
        self.data = data
       
    def animate(self,fps=10,bitrate=1000,dpi = 1000):
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
            p1_vis = x_Log[0:3,i+1]
            v1_vis = x_Log[3:6,i+1].reshape((-1, 1))
            q1_vis = x_Log[6:10,i+1]
            w1_vis = x_Log[10:13,i+1]            

            # -- states robot 2
            p2_vis = x_Log[13:16,i+1]
            v2_vis = x_Log[16:19,i+1].reshape((-1, 1))
            q2_vis = x_Log[19:23,i+1]
            w2_vis = x_Log[23:26,i+1]

            # -- states robot 3
            p3_vis = x_Log[26:29,i+1]
            v3_vis = x_Log[29:32,i+1].reshape((-1, 1))
            q3_vis = x_Log[32:36,i+1]
            w3_vis = x_Log[36:39,i+1]

            # -- states Load
            pL_vis = x_Log[39:42,i+1]
            vL_vis = x_Log[42:45,i+1].reshape((-1, 1))
            qL_vis = x_Log[45:49,i+1]
            wL_vis = x_Log[49:52,i+1]

            # -- states connection
            p1_prime_vis = np.array(p1_vis + dcm_e2b(q1_vis).T @ self.model.r1)
            s1_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ self.model.R1)
            p2_prime_vis = np.array(p2_vis + dcm_e2b(q2_vis).T @ self.model.r2)
            s2_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ self.model.R2)
            p3_prime_vis = np.array(p3_vis + dcm_e2b(q3_vis).T @ self.model.r3)
            s3_vis = np.array(pL_vis + dcm_e2b(qL_vis).T @ self.model.R3)

            # -- inputs --
            F1_vis = np.array(dcm_e2b(q1_vis).T @ self.model.D1 @ u_Log[0:6,i])          
            T1_vis = dcm_e2b(q1_vis).T @ self.model.L1 @ u_Log[0:6,i]                  
            F2_vis = np.array(dcm_e2b(q2_vis).T @ self.model.D2 @ u_Log[6:12,i])                 
            T2_vis = dcm_e2b(q2_vis).T @ self.model.L2 @ u_Log[6:12,i] 
            F3_vis = np.array(dcm_e2b(q3_vis).T @ self.model.D3 @ u_Log[12:18,i])                 
            T3_vis = dcm_e2b(q3_vis).T @ self.model.L3 @ u_Log[12:18,i] 
        
            # -- cable tension
            Tv1_vis = self.model.tension1(xk = x_Log[:,i+1], uk = u_Log[:,i])['Tv1']
            Tv2_vis = self.model.tension2(xk = x_Log[:,i+1], uk = u_Log[:,i])['Tv2']
            Tv3_vis = self.model.tension3(xk = x_Log[:,i+1], uk = u_Log[:,i])['Tv3']

            # -- cable length
            l1 = np.sqrt((p1_prime_vis - s1_vis).T @ (p1_prime_vis - s1_vis))
            l2 = np.sqrt((p2_prime_vis - s2_vis).T @ (p2_prime_vis - s2_vis))
            l3 = np.sqrt((p3_prime_vis - s3_vis).T @ (p3_prime_vis - s3_vis))

            # -- visualization
            ax.clear()
            # -- agents
            plot_robot(p1_vis,q1_vis,ax,clr='red',cle='darkred')
            plot_robot(p2_vis,q2_vis,ax,clr='blue',cle='darkblue')
            plot_robot(p3_vis,q3_vis,ax,clr='violet',cle='darkviolet')
            plot_load(pL_vis,qL_vis,ax)
            plot_cable(p1_prime_vis,s1_vis,ax)
            plot_cable(p2_prime_vis,s2_vis,ax)
            plot_cable(p3_prime_vis,s3_vis,ax)
            # -- force_torque
            plot_vector(p1_vis,F1_vis,ax,scale=0.25)
            plot_vector(p2_vis,F2_vis,ax,scale=0.25)
            plot_vector(p3_vis,F3_vis,ax,scale=0.25)
            plot_vector(p1_vis,T1_vis,ax,clr='c',scale=0.25)
            plot_vector(p2_vis,T2_vis,ax,clr='c',scale=0.25)
            plot_vector(p3_vis,T3_vis,ax,clr='c',scale=0.25)
            # -- tension
            plot_vector(p1_prime_vis,-Tv1_vis,ax,clr='lime',scale=0.1)
            plot_vector(s1_vis,Tv1_vis,ax,clr='lime',scale=0.1)
            plot_vector(p2_prime_vis,-Tv2_vis,ax,clr='lime',scale=0.1)
            plot_vector(s2_vis,Tv2_vis,ax,clr='lime',scale=0.1)
            plot_vector(p3_prime_vis,-Tv3_vis,ax,clr='lime',scale=0.1)
            plot_vector(s3_vis,Tv3_vis,ax,clr='lime',scale=0.1)
            # -- velocity
            plot_vector(p1_vis,v1_vis,ax,clr='orange',scale=0.25)
            plot_vector(p2_vis,v2_vis,ax,clr='orange',scale=0.25)
            plot_vector(p3_vis,v3_vis,ax,clr='orange',scale=0.25)
            plot_vector(pL_vis,vL_vis,ax,clr='orange',scale=0.25)
            for point in range(np.shape(xref_array)[1]):
                xref = xref_array[:,[point]]
                plot_path(xref,ax, alpha = 0.7,THREE=True)

            # -- trajectory
            ax.plot3D(x_Log[39,:],x_Log[40,:],x_Log[41,:],color='maroon', linestyle='dashed')
            ax.set_proj_type('ortho') 
            ax.set_xlim([-5,10])
            ax.set_ylim([-8,8])
            ax.set_zlim([-5,5])
            ax.set_aspect('auto')
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.set_title("Centralized Control",fontsize=20)
            ax.grid()                
            return ax

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=20, azim=-40)

        # Create the animation
        fig.canvas.mpl_connect('key_press_event', toggle_animation)
        self.anim = animation.FuncAnimation(fig, update_frame, frames = np.arange(0, self.n_sim), interval=10) # interval = Delay between frames in milliseconds.
        plt.show()
    
    def plot_input(self,name):
        if name == 'transporter1':
            num = 0
        elif name == 'transporter2':
            num = 1
        elif name == 'transporter3':
            num = 2
        ulim = self.controller.ulim
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
  
    def plot_cost(self):
        cost_Log = self.data['cost']
        time_Log = self.data['compute_time']
        t_cost = np.linspace(0,self.t_sim-self.dt,self.n_sim)

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
    
    def plot_error(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim+1)
        err_x_Log = self.data['err_x'] 
        err_q_Log = self.data['err_q']
        trigger_Log = self.data['trigger']

        # -- position
        plt.rcParams['figure.figsize'] = [6, 8]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(4,1)
        # fig.suptitle("Tracking",fontsize = 22)

        axs[0].plot(t_x,err_x_Log[0,:],color = 'red', linewidth=2)
        axs[0].set_ylabel("error x")

        axs[1].plot(t_x,err_x_Log[1,:],color = 'green', linewidth=2)
        axs[1].set_ylabel("error y")

        axs[2].plot(t_x,err_x_Log[2,:],color = 'blue', linewidth=2)
        axs[2].set_ylabel("error z")

        axs[3].plot(t_x,np.reshape(err_q_Log,(-1,1)),color = 'purple', linewidth=2)
        axs[3].set_ylabel("norm error quaternion")

        for i in range(4):
                axs[i].set_xlim([0,self.t_sim])
                axs[i].grid()
                axs[i].ticklabel_format(useOffset=False)
                for trig in range(trigger_Log.shape[0]):
                    axs[i].axvline(x=trigger_Log[trig], color='indigo', linestyle='--', linewidth=1)
        fig.tight_layout()

    def plot_tension(self):
        # -- extract data
        t_x = np.linspace(0,self.t_sim,self.n_sim)
        direct_Log = self.data['direct'] 

        plt.rcParams['figure.figsize'] = [6, 3]
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(3,1)
        # fig.suptitle("Cable Tension",fontsize = 16)

        axs[0].plot(t_x,direct_Log[0,:],color = 'darkorange')
        axs[0].set_ylabel("cable 1")        
        axs[0].axhline(y=0, color='k', linestyle='--', linewidth=1)

        axs[1].plot(t_x,direct_Log[1,:],color = 'darkorange')
        axs[1].set_ylabel("cable 2")    
        axs[1].axhline(y=0, color='k', linestyle='--', linewidth=1)

        axs[2].plot(t_x,direct_Log[1,:],color = 'darkorange')
        axs[2].set_ylabel("cable 3")    
        axs[2].axhline(y=0, color='k', linestyle='--', linewidth=1)

        for i in range(3):
            axs[i].set_xlim([0,self.t_sim])
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
        fig.tight_layout()
        plt.show()



