#!/usr/bin/env python3

# import ROS2 package
import rclpy
from discower_transportation.node_controller import ControllerNode

# import necessary packages
import numpy as np
import casadi as ca
from transporter.utils.function import *
from transporter.models.centralize_model_2D import CentralizedModel2D
from transporter.controllers.nonlinear_mpc_actual import NonlinearMPCOptiActual
from transporter.controllers.nonlinear_mpc_2D import NonlinearMPCOpti2D
from transporter.utils.plotter import Plotter
from transporter.utils.path_generator import PathGenerator
import pickle
from datetime import datetime


def run(model,controller,init,simulation_time):
    rclpy.init()
    node = ControllerNode(model,controller,init,simulation_time,CONTROLLER_ENABLE=False,PWM_ENABLE=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Stop the controller")
    finally:
        node.get_logger().info("Node is shutting down.")
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()
        return node.data
    


if __name__ == '__main__':
    transporter = CentralizedModel2D()
    dt = 0.1                                              # controller dt
    N = 15*dt                                             # prediction horizon [s]
    Q = ca.diag([100,100,75,75,30,30,
             100,100,75,75,30,30,
             1000,1000,750,750,700,500])*5
    R = ca.diag([1,1,1,1,
                1,1,1,1])*10
    P = Q*10

    ulim = 1.4
    
    controller = NonlinearMPCOpti2D(transporter,N,Q,P,R,dt,ulim=ulim,lambda_angle = 0,
                                lambda_reg = 0,LINEARIZED=False,REDUCED=True)

    # -- initial states Load
    pL_test = ca.DM([[0],[0]])
    theL_test = ca.DM([0])

    # -- initial state robot depends on load at equilibrium
    p1_test,p2_test,the1_test,the2_test = transporter.get_equi_pos_2D(pL_test,theL_test)

    # -- all derivatives is zero
    v1_test  = ca.DM([[0],[0]])
    w1_test  = ca.DM([[0]])
    v2_test  = ca.DM([[0],[0]])
    w2_test  = ca.DM([[0]])
    vL_test  = ca.DM([[0],[0]])
    wL_test  = ca.DM([[0]])

    # -- construct initialize vector
    x0 = ca.vertcat(p1_test,v1_test,the1_test,w1_test,p2_test,v2_test,the2_test,w2_test,pL_test,vL_test,theL_test,wL_test)

    # -- create initial guess for the solver
    x_warm = ca.repmat(x0,1,controller.Nt+1)
    u_warm = np.zeros([transporter.m,controller.Nt])

    ####################################################################################
    ####                            POINT TRACKING                                  #### 
    ####################################################################################
    
    # -- Task 1: Translation in x-direction
    pL_ref = ca.DM([[0.2],[0.0]])
    theL_ref = ca.DM([0])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref1 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- Task 2: Translation in y-direction
    pL_ref = ca.DM([[0.0],[0.2]])
    theL_ref = ca.DM([0])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref2 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- Task 3: Rotation in z-direction
    pL_ref = ca.DM([[0.0],[0.0]])
    theL_ref = ca.DM([np.pi/2])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref3 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- Task 4: Combined task
    pL_ref = ca.DM([[0.2],[0.2]])
    theL_ref = ca.DM([np.pi/12])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref4 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- Task 5: Test
    pL_ref = ca.DM([[0.2],[0.2]])
    theL_ref = ca.DM([0])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref5 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    xref_array = xref4

    ####################################################################################
    ####                           Sequence of POINT TRACKING                       #### 
    ####################################################################################

    # -- translation x task
    pL_ref = ca.DM([[0.2],[0]])
    theL_ref = ca.DM([0])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref1 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- translation y task
    pL_ref = ca.DM([[0.2],[0.2]])
    theL_ref = ca.DM([0])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref2 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- rotate task
    pL_ref = ca.DM([[0.2],[0.2]])
    theL_ref = ca.DM([np.pi/12])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref3 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)

    # -- combining task
    pL_ref = ca.DM([[0.4],[0.0]])
    theL_ref = ca.DM([0])
    p1_ref,p2_ref,the1_ref,the2_ref = transporter.get_equi_pos_2D(pL_ref,theL_ref)
    xref4 = ca.vertcat(p1_ref,0,0,the1_ref,0,p2_ref,0,0,the2_ref,0,pL_ref,0,0,theL_ref,0)
    
    xref_array = np.hstack((xref1,xref2,xref3,xref4))

    ####################################################################################
    ####                            TRAJECTORY TRACKING                             #### 
    ####################################################################################
    path = PathGenerator(transporter)
    # -- circle
    # xref_array = path.generate_circle([-0.5,0],0.5,40,loop_time=100)
    # # -- eight figure
    xref_array = path.generate_eight_figure(1,90,loop_time=100)


    ####################################################################################
    ####                            START CONTROLLER                                #### 
    ####################################################################################
    init = {'x0':x0,'x_warm':x_warm,'u_warm':u_warm,'xref':xref_array}  
    simulation_time = 150

    data = run(transporter,controller,init,simulation_time)
    plotter = Plotter(data,transporter)
    now = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    file_path_gazebo_cen = ''
    with open(file_path_gazebo_cen, 'wb') as file:
        pickle.dump(data, file)

    plotter.plot_all()


