import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from transporter.utils.function import *

class NonlinearMPC3ADecen:
    '''
    Nonlinear MPC Framework for decentralized controller for three agents in 3D manuevering
    '''
    def __init__(self,model,N,Q,P,R,ulim=10, REDUCED = True) -> None:
        self.model = model
        self.dt = self.model.dt
        self.Nt = int(N/self.dt)
        self.Q = Q
        self.P = P
        self.R = R
        self.ulim = ulim
        self.Nx = 26
        self.Nu = 6
        self.set_cost_functions()
        self.create_solver(REDUCED)
    
    def create_solver(self,REDUCED):
        build_solver_time = -time.time()
        self.opti = ca.Opti()                              # create optimization
        self.X = self.opti.variable(self.Nx,self.Nt+1)     # states   
        self.U = self.opti.variable(self.Nu,self.Nt)       # inputs
        self.X0 = self.opti.parameter(self.Nx,1)           # initial states (parameters)
        self.Xref = self.opti.parameter(self.Nx,1)         # reference point (parameters)
        if REDUCED:
            self.lg_multiplier = self.opti.variable(1,self.Nt)
            # self.lg_multiplier_cost = self.opti.variable(1)
        objective = 0


        # -- gap-closing multiple shooting constraints
        for i in range(self.Nt):
            if REDUCED:
                eps = 1e-5
                # self.opti.subject_to(self.X[:,i+1] == self.model.feulctrl(self.X[:,i], self.U[:,i], self.lg_multiplier[:,i]))
                self.opti.subject_to(self.X[:,i+1] == self.model.rk4ctrl(self.X[:,i], self.U[:,i], self.lg_multiplier[:,i]))
                self.opti.subject_to(self.opti.bounded(-eps,self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]), eps))
            else:
                self.opti.subject_to(self.X[:,i+1] == self.model.rk4(self.X[:,i], self.U[:,i])) 
            objective += self.running_cost(self.X[:,i],self.Xref,self.Q,self.U[:,i],self.R)

            # -- constraints maximum input
            self.opti.subject_to(self.opti.bounded(-self.ulim,self.U[:,i],self.ulim)) 

            # -- constraints force to be on one half that tension the cable
            F1 = self.model.D1 @ self.U[0:6,i]
            F1b = dcm_e2b(self.X[6:10,i]).T @ F1 
            self.opti.subject_to(self.model.ctrln1F(self.X[:,i], self.U[:,i]).T @ F1b >= 0)

        objective += self.terminal_cost(self.X[:,self.Nt],self.Xref,self.P)

        # -- initial constraints
        self.opti.subject_to(self.X[:,0] == self.X0)

        # -- objective function
        self.opti.minimize(objective)

        # solver option
        self.sol_options_ipopt = {'ipopt.print_level': 0, 
                             'print_time': 0, 
                             'ipopt.sb': 'yes',
                             'ipopt.warm_start_init_point' : 'yes',
                             'ipopt.max_iter': 10000,
                             'expand':True}
        self.opti.solver('ipopt',self.sol_options_ipopt)
        build_solver_time += time.time()

        print('--------------------Initialize controller--------------------')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print("Number of decision variables: ", self.opti.nx)
        print("Number of parameters: ", self.opti.np)
        print("Number of constraint: ", self.opti.ng)
    
    def solve_mpc(self,x0,xref,x_warm,u_warm,verbose = False):
        solve_time = -time.time()

        # -- set initial value and reference target
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.Xref, xref)

        # -- set initial guess for the solver
        self.opti.set_initial(self.X, x_warm)
        self.opti.set_initial(self.U, u_warm)
        sol = self.opti.solve()
        
        solve_time += time.time()
        if verbose:
            print('# Compute mpc time: %f sec' % solve_time)
        return np.reshape(sol.value(self.U)[:,0],(-1,1)),sol.value(self.X),sol.value(self.U)
    
    def set_cost_functions(self):
        """
        Helper function to setup the cost functions.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx-6, self.Nx-6) # quaternion use only one error, 3 per agent = 12 in total including load
        P = ca.MX.sym('P', self.Nx-6, self.Nx-6)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nx,1)
        xr = ca.MX.sym('xr', self.Nx,1)
        u = ca.MX.sym('u', self.Nu,1)

        # Calculate errors
        err_pv1 = x[0:6] - xr[0:6]
        err_q1 = quatdist(x[6:10],xr[6:10])
        err_w1 = x[10:13] - xr[10:13]

        err_pvL = x[13:19] - xr[13:19]
        err_qL = quatdist(x[19:23],xr[19:23])
        err_wL = x[23:26] - xr[23:26]

        err_vec = ca.vertcat(*[err_pv1,err_q1,err_w1,
                               err_pvL,err_qL,err_wL])

        # Calculate running cost
        ln = err_vec.T @ Q @ err_vec + u.T @ R @ u
        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        V = err_vec.T @ P @ err_vec 
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

