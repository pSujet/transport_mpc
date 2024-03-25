import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
import casadi.tools as ctools
from transporter.utils.function import *

class NonlinearMPC:
    '''
    Nonlinear MPC Framework
    '''
    def __init__(self,model,N,Q,P,R) -> None:
        self.model = model
        self.dt = self.model.dt
        self.Nt = int(N/self.dt)
        self.Q = Q
        self.P = P
        self.R = R
        self.Nx = model.n
        self.Nu = model.m
        self.set_cost_functions()
        self.create_solver()
    
    def create_solver(self):
        build_solver_time = -time.time()
        optvar = ctools.struct_symMX([ctools.entry('U', shape = (self.Nu,1),repeat = self.Nt),
                                        ctools.entry('X', shape = (self.Nx,1), repeat = self.Nt+1)])
        X0 = ca.MX.sym('X0', self.Nx,1)
        Xref = ca.MX.sym('Xref', self.Nx,1)
        param = ca.vertcat(X0,Xref)
        self.optvar = optvar
        self.num_var = optvar.size

        # -- Decision variable boundries
        self.optvar_lb = optvar(-np.inf)
        self.optvar_ub = optvar(np.inf)

        # -- Set initial values
        objective = 0
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []

        # -- initial constraint
        con_eq.append(optvar['X',0] - X0)

        # -- gap-closing multiple shooting constraints
        for i in range(self.Nt):
            x_t = optvar['X',i]
            u_t = optvar['U',i]
            x_ref = Xref
            x_t_next = self.model.rk4(optvar['X',i],optvar['U',i])
            con_eq.append(x_t_next - optvar['X',i+1])

            # Objective Function / Cost Function
            objective += self.running_cost(x_t, x_ref, self.Q, u_t, self.R)
        
        # Terminal cost
        objective += self.terminal_cost(x_t, x_ref, self.P)

        # number of constraints
        # num_eq_con = len(con_eq)
        # num_ineq_con = len(con_ineq) 
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()

        # boundary for equality constraints are zero
        con_eq_lb = np.zeros((num_eq_con, 1))
        con_eq_ub = np.zeros((num_eq_con, 1))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)
        nlp = dict(x = optvar, f = objective, g = con, p = param)

        # solver option
        self.sol_options_ipopt = {'ipopt.print_level': 0, 
                             'print_time': 0, 
                             'ipopt.sb': 'yes',
                             'ipopt.warm_start_init_point' : 'yes',
                             'expand':True}

        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, self.sol_options_ipopt)
        build_solver_time += time.time()
        print('--------------------Initialize controller--------------------')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Receding horizon length: %d ' % self.Nt)        
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
    
    def solve_mpc(self,x0,xref,x_warm):
        param = np.vstack([x0,xref])
        solver_args = dict(x0 = x_warm,
                    lbx = self.optvar_lb,
                    ubx = self.optvar_ub,
                    lbg = self.con_lb,
                    ubg = self.con_ub,
                    p = param)
        solve_time = -time.time()
        sol = self.solver(**solver_args)
        solve_time += time.time()
        optvar = self.optvar(sol['x'])
        self.solve_time = solve_time

        print('# Compute mpc time: %f sec' % solve_time)

        return optvar['U',0],optvar['X']
    
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

        # Calculate errors
        err = x - xr
        err_vec = ca.vertcat(*[err])

        # Calculate running cost
        ln = err_vec.T @ Q @ err_vec + u.T @ R @ u
        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        V = err_vec.T @ P @ err_vec 
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

