#!/usr/bin/env python3

# import ROS2 package
import rclpy
from discower_transportation.node_decen_controller import ControllerDecenNode

# import necessary packages
import numpy as np
import casadi as ca
from transporter.utils.function import *
from transporter.models.centralize_model_2D import CentralizedModel2D
from transporter.models.decentralize_model_2D import DecentralizedModel2D
from transporter.controllers.nonlinear_mpc_2D_decen import NonlinearMPC2DDec
from transporter.utils.plotter import Plotter
from transporter.utils.path_generator import PathGenerator
import argparse
import pickle
from datetime import datetime

def run(model,controller,init,simulation_time,agent):
    rclpy.init()
    node = ControllerDecenNode(model,controller,init,simulation_time,CONTROLLER_ENABLE=False,PWM_ENABLE=True,agent=agent)
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

def read_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=int, help='Agent number')
    args = parser.parse_args()
    if args.agent is not None:
        return args.agent
    else:
        print("Agent argument is missing.")    

if __name__ == '__main__':
    agent = read_arg()                                      # get agent number

    r1 = ca.DM([[0.0],[-0.2]])                                      # vector from CM of robot 1 to anchor point 
    r2 = ca.DM([[0.0],[0.2]])                                       # vector from CM of robot 2 to anchor point 
    R1 = ca.DM([[0.0],[0.2]])                                       # vector from CM of load to anchor point 1
    R2 = ca.DM([[0.0],[-0.2]])                                      # vector from CM of load to anchor point 2
    transporter = CentralizedModel2D()
    if agent == 1:
        transporter_decen = DecentralizedModel2D(rdc=r1,Rdc1=R1,Rdc2=R2)
    elif agent == 2:
        transporter_decen = DecentralizedModel2D(rdc=r2,Rdc1=R2,Rdc2=R1)
    dt = 0.1                                              # controller dt
    N = 15*dt                                             # prediction horizon [s]   
    Q = ca.diag([100,100,50,50,80,50,
             1500,1500,800,800,2000,300])*10
    R = ca.diag([1,1,1,1])*50
    P = Q*10


    controller = NonlinearMPC2DDec(transporter_decen,N,Q,P,R,dt,lambda_reg=0,LINEARIZED=False,REDUCED=True,GUESS=False)
    
    # -- initial states Load
    pL_test = ca.DM([[0],[0]])
    theL_test = ca.DM([0])

    # -- initial state robot depends on load at equilibrium
    p1_test,p2_test,the1_test,the2_test = transporter.get_equi_pos_2D(pL_test,theL_test)

    # -- all derivatives is zero and contruct initialize vector
    x0 = ca.vertcat(p1_test,0,0,the1_test,0,p2_test,0,0,the2_test,0,pL_test,0,0,theL_test,0)

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
    theL_ref = ca.DM([np.pi/6])
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

    xref_array = xref3

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
    # xref_array = path.generate_circle([-0.5,0],0.5,60,loop_time=100)
    # -- eight figure
    # xref_array = path.generate_eight_figure(2,100,loop_time=150)


    ####################################################################################
    ####                            START CONTROLLER                                #### 
    ####################################################################################
    init = {'x0':x0,'x_warm':x_warm,'u_warm':u_warm,'xref':xref_array}  
    simulation_time = 150

    data = run(transporter,controller,init,simulation_time,agent)
    plotter = Plotter(data,transporter)

    now = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    if agent == 1: 
        # file_path_gazebo_decen = ''
        # with open(file_path_gazebo_decen, 'wb') as file:
        #     pickle.dump(data, file)
        plotter.plot_all()
    elif agent == 2:        
        # file_path_gazebo_decen = ''
        # with open(file_path_gazebo_decen, 'wb') as file:
        #     pickle.dump(data, file)
        # plotter.plot_input('transporter2')
        pass
