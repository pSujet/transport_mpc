import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from transporter.utils.function import *

class NonlinearMPCOpti3A:
    '''
    Nonlinear MPC Framework for three agents in 3D manuevering
    '''
    def __init__(self,model,N,Q,P,R,ulim = 10, REDUCED = False) -> None:
        self.model = model
        self.dt = self.model.dt
        self.Nt = int(N/self.dt)
        self.Q = Q
        self.P = P
        self.R = R
        self.ulim = ulim
        self.Nx = model.n
        self.Nu = model.m
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
            self.lg_multiplier = self.opti.variable(3,self.Nt)
        objective = 0


        # -- gap-closing multiple shooting constraints
        for i in range(self.Nt):
            if REDUCED:
                eps = 1e-5
                self.opti.subject_to(self.opti.bounded(-eps,self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]), eps))
                self.opti.subject_to(self.opti.bounded(-eps,self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i]), eps))
                self.opti.subject_to(self.opti.bounded(-eps,self.model.psi3ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[2,i]), eps))
                
                self.opti.subject_to(self.X[:,i+1] == self.model.rk4ctrl(self.X[:,i], self.U[:,i], self.lg_multiplier[:,i]))
            else:
                self.opti.subject_to(self.X[:,i+1] == self.model.rk4(self.X[:,i], self.U[:,i])) 
            objective += self.running_cost(self.X[:,i],self.Xref,self.Q,self.U[:,i],self.R)

            # -- constraints maximum input
            self.opti.subject_to(self.opti.bounded(-self.ulim,self.U[:,i],self.ulim)) 

            # -- constraints force to be on one half that tension the cable
            F1 = self.model.D1 @ self.U[0:6,i]
            F2 = self.model.D2 @ self.U[6:12,i]
            F3 = self.model.D3 @ self.U[12:18,i]
            F1b = dcm_e2b(self.X[6:10,i]).T @ F1 
            F2b = dcm_e2b(self.X[19:23,i]).T @ F2 
            F3b = dcm_e2b(self.X[32:36,i]).T @ F3 
            N1 = - self.model.n1F(self.X[:,i], self.U[:,i])
            N2 = - self.model.n2F(self.X[:,i], self.U[:,i])
            N3 = - self.model.n3F(self.X[:,i], self.U[:,i])
            self.opti.subject_to(self.model.n1F(self.X[:,i], self.U[:,i]).T @ F1b >= 0)
            self.opti.subject_to(self.model.n2F(self.X[:,i], self.U[:,i]).T @ F2b >= 0)
            self.opti.subject_to(self.model.n3F(self.X[:,i], self.U[:,i]).T @ F3b >= 0)

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
        Q = ca.MX.sym('Q', self.Nx-12, self.Nx-12) # quaternion use only one error 3 per agent = 12 in total including load
        P = ca.MX.sym('P', self.Nx-12, self.Nx-12)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nx,1)
        xr = ca.MX.sym('xr', self.Nx,1)
        u = ca.MX.sym('u', self.Nu,1)

        # Calculate errors
        err_pv1 = x[0:6] - xr[0:6]
        err_q1 = quatdist(x[6:10],xr[6:10])
        err_w1 = x[10:13] - xr[10:13]

        err_pv2 = x[13:19] - xr[13:19]
        err_q2 = quatdist(x[19:23],xr[19:23])
        err_w2 = x[23:26] - xr[23:26]

        err_pv3 = x[26:32] - xr[26:32]
        err_q3 = quatdist(x[32:36],xr[32:36])
        err_w3 = x[36:39] - xr[36:39]

        err_pvL = x[39:45] - xr[39:45]
        err_qL = quatdist(x[45:49],xr[45:49])
        err_wL = x[49:52] - xr[49:52]

        err_vec = ca.vertcat(*[err_pv1,err_q1,err_w1,
                               err_pv2,err_q2,err_w2,
                               err_pv3,err_q3,err_w3,
                               err_pvL,err_qL,err_wL])

        # Calculate running cost
        ln = err_vec.T @ Q @ err_vec + u.T @ R @ u
        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        V = err_vec.T @ P @ err_vec 
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

