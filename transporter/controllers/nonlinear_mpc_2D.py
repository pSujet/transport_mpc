import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from transporter.utils.function import *

class NonlinearMPCOpti2D:
    '''
    Nonlinear MPC Framework with Opti
    '''
    def __init__(self,model,N,Q,P,R,dt,ulim = 1.4,lambda_angle = 0,lambda_reg = 0,LINEARIZED = False, REDUCED = False) -> None:
        self.model = model
        self.dt = dt
        self.Nt = int(N/self.dt)
        self.Q = Q
        self.P = P
        self.R = R
        self.ulim = ulim                                    # maximum input [N]
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
            if LINEARIZED:
                # -- linearized discretized
                # self.opti.subject_to(self.X[:,i+1] == self.model.rk4_lin(self.X[:,i], self.U[:,i],self.Ad,self.Bd))
                
                # -- discretized linearized 
                self.opti.subject_to(self.X[:,i+1] == self.Ad @ self.X[:,i] + self.Bd @ self.U[:,i])
                # self.opti.subject_to(self.X[:,i+1] == self.model.Ad @ self.X[:,i] + self.model.Bd @ self.U[:,i])
            else:
                if REDUCED:
                    # self.opti.subject_to(self.X[:,i+1] == self.model.rk4ctrl(self.X[:,i], self.U[:,i],self.dt, self.lg_multiplier[:,i]))
                    self.opti.subject_to(self.X[:,i+1] == self.model.feulctrl(self.X[:,i], self.U[:,i],self.dt, self.lg_multiplier[:,i]))
                    # -- cost
                    # self.opti.subject_to(self.opti.bounded(-self.lg_multiplier_cost,self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]), self.lg_multiplier_cost))
                    # self.opti.subject_to(self.opti.bounded(-self.lg_multiplier_cost,self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i]), self.lg_multiplier_cost))
                    # objective += self.lambda_reg * self.lg_multiplier_cost**2
                    # -- small value
                    eps = 1e-5
                    self.opti.subject_to(self.opti.bounded(-eps,self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i]), eps))
                    self.opti.subject_to(self.opti.bounded(-eps,self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i]), eps))
                    # self.opti.subject_to(self.model.psi1ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[0,i])==0)
                    # self.opti.subject_to(self.model.psi2ddotF(self.X[:,i], self.U[:,i],self.lg_multiplier[1,i])==0)
                else:
                    self.opti.subject_to(self.X[:,i+1] == self.model.rk4(self.X[:,i], self.U[:,i])) 
            objective += self.running_cost(self.X[:,i],self.Xref,self.Q,self.U[:,i],self.R)
            objective += -self.model.dist1F(self.X[:,i], self.U[:,i])*self.lambda_reg#*(self.X[14:16,i].T@self.X[14:16,i])
            objective += -self.model.dist2F(self.X[:,i], self.U[:,i])*self.lambda_reg#*(self.X[14:16,i].T@self.X[14:16,i])

            # -- constraints maximum input
            self.opti.subject_to(self.opti.bounded(-self.ulim,self.U[:,i],self.ulim)) 

            # -- constraints force to be on one half that tension the cable
            F1 = self.model.D1 @ self.U[0:4,i]
            F2 = self.model.D2 @ self.U[4:8,i]
            F1b = dcm_e2b2D(self.X[4,i]).T @ F1 
            F2b = dcm_e2b2D(self.X[10,i]).T @ F2 
            self.opti.subject_to(self.model.n1F(self.X[:,i], self.U[:,i]).T @ F1b >= 0)
            self.opti.subject_to(self.model.n2F(self.X[:,i], self.U[:,i]).T @ F2b >= 0)

            # -- constraints force for tension cable
            b = self.model.bF(self.X[:,i], self.U[:,i]).T 
            F_const = 0.2
            self.opti.subject_to(b @ F1b >= F_const)
            self.opti.subject_to(-b @ F2b >= F_const)

            # -- constraints force for tension cable with lagrange multiplier
            # F_const = 0.2
            # self.opti.subject_to(self.lg_multiplier[:,i] >= F_const)
            


            # squared directional force
            # constratint_angle = np.cos(np.deg2rad(20))
            # self.opti.subject_to((self.model.n1F(self.X[:,i], self.U[:,i]).T @ F1b)**2 >= (constratint_angle)**2*(F1b.T@F1b))
            # self.opti.subject_to((self.model.n2F(self.X[:,i], self.U[:,i]).T @ F2b)**2 >= (constratint_angle)**2*(F2b.T@F2b))

        objective += self.terminal_cost(self.X[:,self.Nt],self.Xref,self.P)

        # -- initial constraints
        self.opti.subject_to(self.X[:,0] == self.X0)

        # -- objective function
        self.opti.minimize(objective)

        # -- solver option
        if LINEARIZED:
            qp_opts = {
                'max_iter': 10,
                'error_on_fail': False,
                'print_header': False,
                'print_iter': False
            }
            self.sol_options_sqp = {
                # 'max_iter': 3,
                'qpsol': 'qrqp',
                'convexify_margin': 1e-5,
                'print_header': False,
                'print_time': False,
                'print_iteration': False,
                'qpsol_options': qp_opts
            }
            self.opti.solver('sqpmethod',self.sol_options_sqp)
        else:
            # -- ipopt
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
        # -- get linearized matrix Ad and Bd        
        if self.LINEARIZED:  
            x_bar = xref
            u_bar = np.zeros((8, 1))    
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
        # print("Lambda: ", self.opti.debug.value(self.lg_multiplier))
        if verbose:
            print('# Compute mpc time: %f sec' % solve_time)
        return np.reshape(sol.value(self.U)[:,0],(-1,1)),sol.value(self.X),sol.value(self.U)
    
    def set_cost_functions(self):
        """
        Helper function to setup the cost functions.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx, self.Nx)
        P = ca.MX.sym('P', self.Nx, self.Nx)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nx,1)
        xr = ca.MX.sym('xr', self.Nx,1)
        u = ca.MX.sym('u', self.Nu,1)

        # -- states robot 1
        p1 = x[0:2]
        v1 = x[2:4]
        the1 = x[4]
        w1 = x[5]

        # -- states robot 2
        p2 = x[6:8]
        v2 = x[8:10]
        the2 = x[10]
        w2 = x[11]

        # -- states Load
        pL = x[12:14]
        vL = x[14:16]
        theL = x[16]
        wL = x[17]

        # Calculate errors
        err_pv1 = x[0:4] - xr[0:4]
        # err_the1 = x[4] - xr[4]
        err_the1 = quatdist(yaw2quaternion(x[4]),yaw2quaternion(xr[4]))
        # err_the1 = quatdistexplicit(yaw2quaternion(x[4]),yaw2quaternion(xr[4]))
        err_w1 = x[5] - xr[5]
        err_pv2 = x[6:10] - xr[6:10]
        # err_the2 = x[10] - xr[10]
        err_the2 = quatdist(yaw2quaternion(x[10]),yaw2quaternion(xr[10]))
        # err_the2 = quatdistexplicit(yaw2quaternion(x[10]),yaw2quaternion(xr[10]))
        err_w2 = x[11] - xr[11]
        err_pvL = x[12:16] - xr[12:16]
        # err_theL = x[16] - xr[16]
        err_theL = quatdist(yaw2quaternion(x[16]),yaw2quaternion(xr[16]))
        # err_theL = quatdistexplicit(yaw2quaternion(x[16]),yaw2quaternion(xr[16]))
        err_wL = x[17] - xr[17]
        err_vec = ca.vertcat(*[err_pv1,err_the1,err_w1,
                               err_pv2,err_the2,err_w2,
                               err_pvL,err_theL,err_wL])
        
        # Calculate constant input at equilibrium
        b = self.model.bF(x, u)
        F_const_1 = 0.2 @ b
        F_const_2 = 0.2 @ -b
        u_eq1 = self.model.D1inv @ F_const_1
        u_eq2 = self.model.D2inv @ F_const_2
        u_eq = ca.vertcat(*[u_eq1,u_eq2])

        # Calculate running cost
        # ln = err_vec.T @ Q @ err_vec + u.T @ R @ u
        ln = err_vec.T @ Q @ err_vec + (u-u_eq).T @ R @ (u-u_eq)
        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])
        self.error = ca.Function('err', [x, xr], [err_vec])

        Un = (u-u_eq).T @ R @ (u-u_eq)
        self.input_cost = ca.Function('Un', [x,u, R], [Un])

        # Calculate terminal cost
        V = err_vec.T @ P @ err_vec 
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

