import casadi as ca
import numpy as np

from transporter.models.centralize_model_3A import CentralizedModel3A
from transporter.models.decentralize_model_3A import DecentralizedModel3A
from transporter.controllers.nonlinear_mpc_opti_3A_decen import NonlinearMPC3ADecen
from transporter.simulators.simulation_tracking_3A_decen import Simulator3ADec
from initializer import set_initial

# create the model
r1 = 0.5*ca.DM([[0],[-1],[0]])                                  # vector from CM of robot 1 to anchor point 
r2 = 0.5*ca.DM([[np.cos(np.pi/6)],[np.sin(np.pi/6)],[0]])       # vector from CM of robot 2 to anchor point 
r3 = 0.5*ca.DM([[-np.cos(np.pi/6)],[np.sin(np.pi/6)],[0]])      # vector from CM of robot 2 to anchor point 
R1 = ca.DM([[0],[1],[0]])                                       # vector from CM of Load to anchor point 1
R2 = ca.DM([[-np.sin(np.pi/3)],[-np.cos(np.pi/3)],[0]])         # vector from CM of Load to anchor point 2
R3 = ca.DM([[np.sin(np.pi/3)],[-np.cos(np.pi/3)],[0]])          # vector from CM of Load to anchor point 3

transporter = CentralizedModel3A()                              # for visualization
transporter1 = DecentralizedModel3A(rdc=r1,Rdc1=R1,Rdc2=R2,Rdc3=R3)
transporter2 = DecentralizedModel3A(rdc=r2,Rdc1=R2,Rdc2=R3,Rdc3=R1)
transporter3 = DecentralizedModel3A(rdc=r3,Rdc1=R3,Rdc2=R1,Rdc3=R2)

# MPC parameters for decentralized control
N = 1.5                                             # prediction horizon [s]
Q = ca.diag([20,20,20,10,10,10,20,20,20,20,
             1000,1000,1000,100,100,100,3000,500,500,500])*10
R = ca.diag([1,1,1,1,1,1,])*1
P = Q*10
ulim = 5

# create the controller
controller1 = NonlinearMPC3ADecen(transporter1,N,Q,P,R,ulim=ulim)
controller2 = NonlinearMPC3ADecen(transporter2,N,Q,P,R,ulim=ulim)
controller3 = NonlinearMPC3ADecen(transporter3,N,Q,P,R,ulim=ulim)

# set initial condition
init = set_initial(transporter,controller1,"multi-points")

if __name__ == "__main__":
    sim_time = 15
    sim = Simulator3ADec(transporter,controller1,controller2,controller3)
    sim.run(init,sim_time)
    sim.plot_cost()
    sim.plot_error()    
    sim.animate()
    