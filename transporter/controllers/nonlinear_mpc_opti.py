import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from transporter.utils.function import *

class NonlinearMPCOpti:
    '''
    Nonlinear MPC Framework with Opti
    '''
    def __init__(self,model,N,Q,P,R,lambda_angle = 0,lambda_reg = 0,LINEARIZED = False, REDUCED = False) -> None:
        self.model = model
        self.dt = self.model.dt
        self.Nt = int(N/self.dt)
        self.Q = Q
        self.P = P
        self.R = R
        self.lambda_angle = lambda_angle
        self.lambda_reg = lambda_reg
        self.Nx = model.n
        self.Nu = model.m
        self.LINEARIZED = LINEARIZED
        self.set_cost_functions()
        self.create_solver(LINEARIZED,REDUCED)
    
    def create_solver(self,LINEARIZED,REDUCED):
        build_solver_time = -time.time()
        self.opti = ca.Opti()                              # create optimization
        self.X = self.opti.variable(self.Nx,self.Nt+1)     # states   
        self.U = self.opti.variable(self.Nu,self.Nt)       # inputs
        self.X0 = self.opti.parameter(self.Nx,1)           # initial states (parameters)
        self.Xref = self.opti.parameter(self.Nx,1)         # reference point (parameters)
        if LINEARIZED:
            self.Ad = self.opti.parameter(self.Nx,self.Nx)
            self.Bd = self.opti.parameter(self.Nx,self.Nu)
        if REDUCED:
            self.lg_multiplier = self.opti.variable(2,self.Nt)
            self.lg_multiplier_cost = self.opti.variable(1)
        objective = 0


        # -- gap-closing multiple shooting constraints
        for i in range(self.Nt):
            # -- normalized quaternion 
            # X_temp = self.model.rk4(self.X[:,i], self.U[:,i])            
            # X_temp[6:10] = X_temp[6:10]/ca.sqrt(X_temp[6:10].T @ X_temp[6:10])
            # X_temp[19:23] = X_temp[19:23]/ca.sqrt(X_temp[19:23].T @ X_temp[19:23])
            # X_temp[32:36] = X_temp[32:36]/ca.sqrt(X_temp[32:36].T @ X_temp[32:36])
            # self.opti.subject_to(self.X[:,i+1] == X_temp)

            # -- no normalized quaternion 
            if LINEARIZED:
                # -- linearized discretized
                # self.opti.subject_to(self.X[:,i+1] == self.model.rk4_lin(self.X[:,i], self.U[:,i],self.Ad,self.Bd))
                
                # -- discretized linearized 
                self.opti.subject_to(self.X[:,i+1] == self.Ad @ self.X[:,i] + self.Bd @ self.U[:,i])
                # self.opti.subject_to(self.X[:,i+1] == self.model.Ad @ self.X[:,i] + self.model.Bd @ self.U[:,i])
            else:
                if REDUCED:
                    self.opti.subject_to(self.X[:,i+1] == self.model.rk4ctrl(self.X[:,i], self.U[:,i], self.lg_multiplier[:,i]))
                    # self.opti.subject_to(self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]) == 0)
                    # self.opti.subject_to(self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i]) == 0)
                    # self.opti.subject_to(self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]) <= self.lg_multiplier_cost)
                    # self.opti.subject_to(self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i]) <= self.lg_multiplier_cost)
                    self.opti.subject_to(self.opti.bounded(-self.lg_multiplier_cost,self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]),self.lg_multiplier_cost))
                    self.opti.subject_to(self.opti.bounded(-self.lg_multiplier_cost,self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i]), self.lg_multiplier_cost))
                    objective += self.lambda_reg * self.lg_multiplier_cost**2
                else:
                    self.opti.subject_to(self.X[:,i+1] == self.model.rk4(self.X[:,i], self.U[:,i])) 
            objective += self.running_cost(self.X[:,i],self.Xref,self.Q,self.U[:,i],self.R)

            # -- constraints maximum input
            self.opti.subject_to(self.opti.bounded(-10,self.U[:,i],10)) 

            # -- constraints force to be on one half that tension the cable
            F1 = self.model.D1 @ self.U[0:6,i]
            F2 = self.model.D2 @ self.U[6:12,i]
            F1b = dcm_e2b(self.X[6:10,i]).T @ F1 
            F2b = dcm_e2b(self.X[19:23,i]).T @ F2 
            N1 = - self.model.n1F(self.X[:,i], self.U[:,i])
            N2 = - self.model.n2F(self.X[:,i], self.U[:,i])
            # self.opti.subject_to(self.model.n1F(self.X[:,i], self.U[:,i]).T @ (dcm_e2b(self.X[6:10,i]).T @ F1) >= 0)
            # self.opti.subject_to(self.model.n2F(self.X[:,i], self.U[:,i]).T @ (dcm_e2b(self.X[19:23,i]).T @ F2) >= 0)
            self.opti.subject_to(self.model.n1F(self.X[:,i], self.U[:,i]).T @ F1b >= 0)
            self.opti.subject_to(self.model.n2F(self.X[:,i], self.U[:,i]).T @ F2b >= 0)
            # self.opti.subject_to(N1.T @ N1 + F1b.T @ F1b <= 0)
            # self.opti.subject_to(N2.T @ N2 + F2b.T @ F2b <= 0)

            # -- add soft constraints/cost on angle diff from equilibrium
            if LINEARIZED:
                pass
                # objective += self.lambda_angle*(1-self.model.angle_diff1(self.X[:,i], self.U[:,i]))**2
                # objective += self.lambda_angle*(1-self.model.angle_diff2(self.X[:,i], self.U[:,i]))**2
                # angle_diff_robot1 = self.model.n1F(self.X[:,i], self.U[:,i]).T @ - dcm_e2b(self.X[6:10,i]).T @ self.model.r1
                # angle_diff_robot2 = self.model.n2F(self.X[:,i], self.U[:,i]).T @ - dcm_e2b(self.X[19:23,i]).T @ self.model.r2
                # objective += self.lambda_angle*(1-angle_diff_robot1)**2
                # objective += self.lambda_angle*(1-angle_diff_robot2)**2     
                        
            # -- constraints for linearization
            # angle_limit  = 30 # degree
            # cos_angle_limit = np.cos(angle_limit*np.pi/180)
            # self.opti.subject_to(self.model.n1F(self.X[:,i], self.U[:,i]).T @ self.model.bF(self.X[:,i], self.U[:,i]) >= cos_angle_limit)
            # self.opti.subject_to(-self.model.n2F(self.X[:,i], self.U[:,i]).T @ self.model.bF(self.X[:,i], self.U[:,i]) >= cos_angle_limit)

            # opti.subject_to(self.model.n1F(X[:,i], U[:,i]).T @ (dcm_e2b(X[6:10,i]).T @ self.model.r1) < 0)
            # opti.subject_to(self.model.n2F(X[:,i], U[:,i]).T @ (dcm_e2b(X[19:23,i]).T @ self.model.r2) < 0)
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
    
    def solve_mpc(self,x0,xref,x_warm,u_warm,verbose = False):
        solve_time = -time.time()
        # -- get linearized matrix Ad and Bd
        x_bar = xref
        u_bar = np.zeros((12, 1))        
        Ad = self.model.evalAd(x_bar, u_bar)
        Bd = self.model.evalBd(x_bar, u_bar)

        # -- set initial value and reference target
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.Xref, xref)
        if self.LINEARIZED:
            self.opti.set_value(self.Ad, Ad)
            self.opti.set_value(self.Bd, Bd)

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
        Q = ca.MX.sym('Q', self.Nx-9, self.Nx-9) # quaternion use only one error
        P = ca.MX.sym('P', self.Nx-9, self.Nx-9)
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
        err_pvL = x[26:32] - xr[26:32]
        err_qL = quatdist(x[32:36],xr[32:36])
        err_wL = x[36:39] - xr[36:39]
        err_vec = ca.vertcat(*[err_pv1,err_q1,err_w1,
                               err_pv2,err_q2,err_w2,
                               err_pvL,err_qL,err_wL])

        # Calculate running cost
        ln = err_vec.T @ Q @ err_vec + u.T @ R @ u
        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        V = err_vec.T @ P @ err_vec 
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

