import casadi as ca

from transporter.controllers.nonlinear_mpc_opti_3A import NonlinearMPCOpti3A
from transporter.models.centralize_model_3A import CentralizedModel3A
from transporter.simulators.simulation_tracking_3A import Simulator3A
from initializer import set_initial

# create the model
transporter = CentralizedModel3A()

# MPC parameters for centralized control
N = 1.5                                             # prediction horizon [s]
Q = ca.diag([10,10,10,1,1,1,10,1,1,1,
             10,10,10,1,1,1,10,1,1,1,
             10,10,10,1,1,1,10,1,1,1,
             500,500,500,5,5,5,500,5,5,5])*10
R = ca.diag([1,1,1,1,1,1,
             1,1,1,1,1,1,
             1,1,1,1,1,1])*1
P = Q*10
ulim = 5

# create the controller (set REDUCED to True to use the reduced model)
controller = NonlinearMPCOpti3A(transporter,N,Q,P,R,ulim=ulim,REDUCED=True)

# set initial condition
init = set_initial(transporter,controller,"multi-points")

if __name__ == "__main__":
    sim_time = 15
    sim = Simulator3A(transporter,controller)
    sim.run(init,sim_time)
    sim.plot_cost()
    sim.plot_error()    
    sim.animate()
    
    