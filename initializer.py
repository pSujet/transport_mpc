import casadi as ca
import matplotlib.pyplot as plt
from transporter.utils.function import *
from transporter.controllers.nonlinear_mpc_opti_3A import NonlinearMPCOpti3A
from transporter.models.centralize_model_3A import CentralizedModel3A


def set_initial(transporter,controller,demo):
    if demo == "one-point":
        # initial states of the load
        pL_init = ca.DM([[0],[0],[0]])
        qL_init = ca.DM([[0],[0],[0],[1]])
        qL_init = qL_init/np.linalg.norm(qL_init)

        # all derivatives is zero
        zero_init  = ca.DM([[0],[0],[0]])
        zero_ref = zero_init

        # get state of robots from the pose of the load at equilibrium
        p1_init,p2_init,p3_init,q1_init,q2_init,q3_init = transporter.get_equi_pos(pL_init,qL_init)

        # stack all the states in a single vector
        x0 = ca.vertcat(p1_init,zero_init,q1_init,zero_init,p2_init,zero_init,q2_init,zero_init,p3_init,zero_init,q3_init,zero_init,pL_init,zero_init,qL_init,zero_init)

        # warm start
        x_warm = ca.repmat(x0,1,controller.Nt+1)
        u_warm = np.zeros([transporter.m,controller.Nt])

        # target pose of the load: moving in x,y,z and rotating around all axis at the same time 45 degrees in ZYX order
        pL_ref = ca.DM([[1],[1],[1]])
        qL_ref = ca.DM([[0.191],[0.462],[0.191],[0.845]])
        qL_ref = qL_ref/np.linalg.norm(qL_ref)
        p1_ref,p2_ref,p3_ref,q1_ref,q2_ref,q3_ref = transporter.get_equi_pos(pL_ref,qL_ref)
        xref_array = ca.vertcat(p1_ref,zero_ref,q1_ref,zero_ref,p2_ref,zero_ref,q2_ref,zero_ref,p3_ref,zero_ref,q3_ref,zero_ref,pL_ref,zero_ref,qL_ref,zero_ref)

    elif demo == "multi-points":
        # initial states of the load
        pL_init = ca.DM([[0],[0],[0]])
        qL_init = ca.DM([[0],[0],[0],[1]])
        qL_init = qL_init/np.linalg.norm(qL_init)

        # get state of robots from the pose of the load at equilibrium
        p1_init,p2_init,p3_init,q1_init,q2_init,q3_init = transporter.get_equi_pos(pL_init,qL_init)

        # all derivatives is zero
        zero_init  = ca.DM([[0],[0],[0]])
        zero_ref = zero_init

        # stack all the states in a single vector
        x0 = ca.vertcat(p1_init,zero_init,q1_init,zero_init,p2_init,zero_init,q2_init,zero_init,p3_init,zero_init,q3_init,zero_init,pL_init,zero_init,qL_init,zero_init)

        # warm start
        x_warm = ca.repmat(x0,1,controller.Nt+1)
        u_warm = np.zeros([transporter.m,controller.Nt])       

        # -- task 1
        pL_ref = ca.DM([[3],[0],[1]])
        qL_ref = ca.DM([[0],[0],[0],[1]])
        qL_ref = qL_ref/np.linalg.norm(qL_ref)
        p1_ref,p2_ref,p3_ref,q1_ref,q2_ref,q3_ref = transporter.get_equi_pos(pL_ref,qL_ref)
        xref1 = ca.vertcat(p1_ref,zero_ref,q1_ref,zero_ref,p2_ref,zero_ref,q2_ref,zero_ref,p3_ref,zero_ref,q3_ref,zero_ref,pL_ref,zero_ref,qL_ref,zero_ref)

        # -- task 2
        pL_ref = ca.DM([[3],[1],[1]])
        qL_ref = ca.DM([[0],[0],[0.707],[0.707]])
        qL_ref = qL_ref/np.linalg.norm(qL_ref)
        p1_ref,p2_ref,p3_ref,q1_ref,q2_ref,q3_ref = transporter.get_equi_pos(pL_ref,qL_ref)
        xref2 = ca.vertcat(p1_ref,zero_ref,q1_ref,zero_ref,p2_ref,zero_ref,q2_ref,zero_ref,p3_ref,zero_ref,q3_ref,zero_ref,pL_ref,zero_ref,qL_ref,zero_ref)

        # -- task 3
        pL_ref = ca.DM([[5],[2],[-1]])
        qL_ref = ca.DM([[0],[0.707],[0],[0.707]])
        qL_ref = qL_ref/np.linalg.norm(qL_ref)
        p1_ref,p2_ref,p3_ref,q1_ref,q2_ref,q3_ref = transporter.get_equi_pos(pL_ref,qL_ref)
        xref3 = ca.vertcat(p1_ref,zero_ref,q1_ref,zero_ref,p2_ref,zero_ref,q2_ref,zero_ref,p3_ref,zero_ref,q3_ref,zero_ref,pL_ref,zero_ref,qL_ref,zero_ref)

        xref_array = np.hstack((xref1,xref2,xref3))
            
    
    # initial conditions
    init = {'x0':x0,
            'x_warm':x_warm,
            'u_warm':u_warm,
            'xref':xref_array}
    
    return init

def plot_compare(sim_centralized,sim_centralized_no_opt,sim_decentralized,txlim):
    ##########################################
    ###            Plot Cost               ###
    ##########################################

    # extract Data
    cost_Log_cen = sim_centralized['cost']
    time_Log_cen = sim_centralized['compute_time']
    t_cost_cen = np.linspace(0,sim_centralized['t_sim'],sim_centralized['n_sim'])

    cost_Log_cen_no_opt = sim_centralized_no_opt['cost']
    time_Log_cen_no_opt = sim_centralized_no_opt['compute_time']
    t_cost_cen_no_opt = np.linspace(0,sim_centralized_no_opt['t_sim'],sim_centralized_no_opt['n_sim'])

    cost_Log_decen = sim_decentralized['cost']
    time_Log_decen = sim_decentralized['compute_time']
    t_cost_decen = np.linspace(0,sim_decentralized['t_sim'],sim_decentralized['n_sim'])

    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['figure.dpi'] = 100
    fig, axs = plt.subplots(2,1)

    axs[0].plot(t_cost_cen_no_opt,cost_Log_cen_no_opt[0,:],label='centralized',color = 'dodgerblue',linestyle = '-.')
    axs[0].plot(t_cost_cen,cost_Log_cen[0,:],label='centralized(reduced)',color = 'darkblue')    
    axs[0].plot(t_cost_decen,cost_Log_decen[0,:],label='decentralized',color = 'lightseagreen',linestyle = '--')
    axs[0].set_ylabel("Running Cost")
    axs[0].set_xlabel("Time [s]")
    axs[0].grid()
    axs[0].set_xlim([0,sim_decentralized['t_sim']])
    axs[0].set_xlim([0,txlim])
    axs[0].legend(loc = 'upper right',fontsize = 10)

    axs[1].plot(t_cost_cen_no_opt[1:],time_Log_cen_no_opt[0,1:],label='centralized',marker=".",color = 'dodgerblue',linestyle = '-.')
    axs[1].plot(t_cost_cen[1:],time_Log_cen[0,1:],label='centralized(reduced)',marker=".",color = 'darkblue')    
    axs[1].plot(t_cost_decen[1:],time_Log_decen[0,1:],label='decentralized',marker=".",color = 'lightseagreen',linestyle = '--')
    axs[1].set_ylabel("Computed Time [s]")
    axs[1].set_xlabel("Time [s]")
    axs[1].grid()
    axs[1].set_xlim([0,sim_decentralized['t_sim']])
    axs[1].set_xlim([0,txlim])
    axs[1].axhline(y = np.mean(time_Log_cen[0,1:]), color = 'red', linestyle = '--')
    axs[1].axhline(y = np.mean(time_Log_cen_no_opt[0,1:]), color = 'red', linestyle = '--')
    axs[1].axhline(y = np.mean(time_Log_decen[0,1:]), color = 'red', linestyle = '--')
    axs[1].legend(loc = 'upper right',fontsize = 10)
    fig.tight_layout()

    print("-- Centralized Controller --")
    print("Max initialized computed time: " + str(np.max(time_Log_cen_no_opt))+" s")
    print("Max computed time: " + str(np.max(time_Log_cen_no_opt[0,1:]))+" s")
    print("Average computed time: "+ str(np.mean(time_Log_cen_no_opt[0,1:]))+" s")
    print("-- Centralized Controller (reduced) --")
    print("Max initialized computed time: " + str(np.max(time_Log_cen))+" s")
    print("Max computed time: " + str(np.max(time_Log_cen[0,1:]))+" s")
    print("Average computed time: "+ str(np.mean(time_Log_cen[0,1:]))+" s")
    print("-- Decentralized Controller --")
    print("Max initialized computed time: " + str(np.max(time_Log_decen))+" s")
    print("Max computed time: " + str(np.max(time_Log_decen[0,1:]))+" s")
    print("Average computed time: "+ str(np.mean(time_Log_decen[0,1:]))+" s")

    ##########################################
    ###            Plot State              ###
    ##########################################
    # -- extract data
    t_x_cen = np.linspace(0,sim_centralized['t_sim'],sim_centralized['n_sim']+1)
    err_x_Log_cen = sim_centralized['err_x'] 
    err_q_Log_cen = sim_centralized['err_q']

    t_x_decen = np.linspace(0,sim_decentralized['t_sim'],sim_decentralized['n_sim']+1)
    err_x_Log_decen = sim_decentralized['err_x'] 
    err_q_Log_decen = sim_decentralized['err_q']

    t_x_cen_no_opt = np.linspace(0,sim_centralized_no_opt['t_sim'],sim_centralized_no_opt['n_sim']+1)
    err_x_Log_cen_no_opt = sim_centralized_no_opt['err_x'] 
    err_q_Log_cen_no_opt = sim_centralized_no_opt['err_q']

    # -- position
    plt.rcParams['figure.figsize'] = [6, 8]
    plt.rcParams['figure.dpi'] = 100
    fig, axs = plt.subplots(4,1)
    #plot with color value rgb

    axs[0].plot(t_x_cen_no_opt,err_x_Log_cen_no_opt[0,:],label='centralized',color = 'red', linewidth=2, linestyle = '-.')
    axs[0].plot(t_x_cen,err_x_Log_cen[0,:],label='centralized(reduced)',color = 'darkred', linewidth=2)
    axs[0].plot(t_x_decen,err_x_Log_decen[0,:],label='decentralized',color = 'darkorange', linewidth=2, linestyle = '--')
    axs[0].set_ylabel("error x [m]")

    axs[1].plot(t_x_cen_no_opt,err_x_Log_cen_no_opt[1,:],label='centralized',color = 'green', linewidth=2, linestyle = '-.')
    axs[1].plot(t_x_cen,err_x_Log_cen[1,:],label='centralized(reduced)',color = 'darkgreen', linewidth=2)
    axs[1].plot(t_x_decen,err_x_Log_decen[1,:],label='decentralized',color = 'limegreen', linewidth=2, linestyle = '--')
    axs[1].set_ylabel("error y [m]")

    axs[2].plot(t_x_cen_no_opt,err_x_Log_cen_no_opt[2,:],label='centralized',color = 'blue', linewidth=2, linestyle = '-.')
    axs[2].plot(t_x_cen,err_x_Log_cen[2,:],label='centralized(reduced)',color = 'darkblue', linewidth=2)
    axs[2].plot(t_x_decen,err_x_Log_decen[2,:],label='decentralized',color = 'turquoise', linewidth=2, linestyle = '--')
    axs[2].set_ylabel("error z [m]")

    axs[3].plot(t_x_cen_no_opt,np.reshape(err_q_Log_cen_no_opt,(-1,1)),label='centralized',color = 'magenta', linewidth=2, linestyle = '-.')
    axs[3].plot(t_x_cen,np.reshape(err_q_Log_cen,(-1,1)),label='centralized(reduced)',color = 'darkviolet', linewidth=2)
    axs[3].plot(t_x_decen,np.reshape(err_q_Log_decen,(-1,1)),label='decentralized',color = 'violet', linewidth=2, linestyle = '--')
    axs[3].set_ylabel("norm error quaternion")

    for i in range(4):
            axs[i].set_xlim([0,sim_centralized['t_sim']])
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
            axs[i].legend(loc = 'upper right',fontsize = 10)
    fig.tight_layout()

    end_index = 80
    print("============ Error ============")
    print("-- Centralized Controller --")
    print("Average Position Error: " + str(np.mean(np.sqrt(np.sum(np.square(err_x_Log_cen_no_opt[:,0:end_index]),axis=0)))))
    print("Std Position Error: " + str(np.std(np.sqrt(np.sum(np.square(err_x_Log_cen_no_opt[:,0:end_index]),axis=0)))))
    print("Average Quaternion Error: " + str(np.mean(err_q_Log_cen_no_opt[:,0:end_index])))
    print("Std Quaternion Error: " + str(np.std(err_q_Log_cen_no_opt[:,0:end_index])))
    print("-- Centralized Controller (reduced) --")
    print("Average Position Error: " + str(np.mean(np.sqrt(np.sum(np.square(err_x_Log_cen[:,0:end_index]),axis=0)))))
    print("Std Position Error: " + str(np.std(np.sqrt(np.sum(np.square(err_x_Log_cen[:,0:end_index]),axis=0)))))
    print("Average Quaternion Error: " + str(np.mean(err_q_Log_cen[:,0:end_index])))
    print("Std Quaternion Error: " + str(np.std(err_q_Log_cen[:,0:end_index])))
    print("-- Decentralized Controller --")
    print("Average Position Error: " + str(np.mean(np.sqrt(np.sum(np.square(err_x_Log_decen[:,0:end_index]),axis=0)))))
    print("Std Position Error: " + str(np.std(np.sqrt(np.sum(np.square(err_x_Log_decen[:,0:end_index]),axis=0)))))
    print("Average Quaternion Error: " + str(np.mean(err_q_Log_decen[:,0:end_index])))
    print("Std Quaternion Error: " + str(np.std(err_q_Log_decen[:,0:end_index])))
    plt.show()



