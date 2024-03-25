"""
Nonlinear MPC Framework
Sujet Phodapol
"""

import time
import numpy as np
import casadi as ca
import casadi.tools as ctools

class NonlinearMPC:
    def __init__(self, model, dynamics,Q, P, R, 
                 solver_type='ipopt', horizon=10,
                 **kwargs):
        """
        Nolinear MPC Controller Class for setpoint tracking
        - param model: model class (type: python class)
        - param dynamics: system dynamics function (type: ca.Function)
        - param horizon: prediction horizon [s], defaults to 10s (type: float, optional)
        - param Q: state error weight matrix (type Q: ca.diag)
        - param P: terminal state weight matrix (type P: np.array)
        - param R: control input weight matrix (type R: np.diag)
        - param ulb: control lower bound, defaults to None (type ulb: np.array, optional)
        - param uub: control upper bound, defaults to None (type uub: np.array, optional)
        - param xlb: state lower bound, defaults to None (type xlb: np.array, optional)
        - param xub:state upper bound, defaults to None (type xub: np.array, optional)
        - param terminal_constraint: terminal constraint set, defaults to None (type terminal_constraint: np.array, optional)
        """
        self.solve_time = 0.0
        self.solver_type = solver_type
        self.dt = model.dt
        self.Nx = model.n                           # number of states
        self.Nu = model.m                           # number of inputs
        self.model = model
        self.Nt = int(horizon / self.dt)
        self.dynamics = dynamics

        # Initialize variables
        self.Q = ca.MX(Q)
        self.P = ca.MX(P)
        self.R = ca.MX(R)

        # Initialize barrier variables
        if "xub" in kwargs:
            self.xub = kwargs["xub"]
        if "xlb" in kwargs:
            self.xlb = kwargs["xlb"]
        if "uub" in kwargs:
            self.uub = kwargs["uub"]
        if "ulb" in kwargs:
            self.ulb = kwargs["ulb"]
        if "terminal_constraint" in kwargs:
            self.tc_ub = np.full((self.Nx,), kwargs["terminal_constraint"])
            self.tc_lb = np.full((self.Nx,), -kwargs["terminal_constraint"])

        self.set_options_dicts()
        self.set_cost_functions()
        self.test_cost_functions(Q, R, P)
        self.create_solver()

    def create_solver(self):
        """
        Create the solver object.
        """

        build_solver_time = -time.time()

        # Starting state parameters - add slack here
        x0 = ca.MX.sym('x0', self.Nx)
        x_ref = ca.MX.sym('x_ref', self.Nx * (self.Nt + 1),)
        param_s = ca.vertcat(x0, x_ref)

        # Create optimization variables
        opt_var = ctools.struct_symMX([(
                                      ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                      ctools.entry('x', shape=(self.Nx,), repeat=self.Nt + 1),
                                      )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []

        # initial state constraint
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        for t in range(self.Nt):
            # Get variables
            x_t = opt_var['x', t]
            u_t = opt_var['u', t]
            x_r = x_ref[(t * self.Nx):(t * self.Nx + self.Nx)]
      

            # Dynamics constraint (multiple shooting)
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            if hasattr(self, 'uub'):
                con_ineq.append(u_t)
                con_ineq_ub.append(self.uub)
                con_ineq_lb.append(np.full((self.Nu,), -ca.inf))
            if hasattr(self, 'ulb'):
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((self.Nu,), ca.inf))
                con_ineq_lb.append(self.ulb)

            # State constraints
            if hasattr(self, 'xub'):
                con_ineq.append(x_t)
                con_ineq_ub.append(self.xub)
                con_ineq_lb.append(np.full((self.Nx,), -ca.inf))
            if hasattr(self, 'xlb'):
                con_ineq.append(x_t)
                con_ineq_ub.append(np.full((self.Nx,), ca.inf))
                con_ineq_lb.append(self.xlb)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t, x_r, self.Q, u_t, self.R)

        # Terminal Cost
        obj += self.terminal_cost(opt_var['x', self.Nt],x_ref[self.Nt * self.Nx:], self.P)

        # Terminal contraint
        if hasattr(self, 'tc_lb') and hasattr(self, 'tc_ub'):
            con_ineq.append(opt_var['x', self.Nt] - x_ref[self.Nt * self.Nx:])
            con_ineq_lb.append(self.tc_lb)
            con_ineq_ub.append(self.tc_ub)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = len(con_eq)
        num_ineq_con = len(con_ineq)
        # num_eq_con = ca.vertcat(*con_eq).size1()
        # num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con, 1))
        con_eq_ub = np.zeros((num_eq_con, 1))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)

        # Instantiate solver
        self.set_solver_dictionaries(nlp)
        self.solver = self.solver_dict[self.solver_type]

        build_solver_time += time.time()
        print('\n________________________________________')
        print('Initialize controller')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Receding horizon length: %d ' % self.Nt)        
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass

    def set_options_dicts(self):
        """
        Helper function to set the dictionaries for solver and function options
        """

        # Functions options
        self.fun_options = {}

        # Options for NLP Solvers
        # -> SQP Method
        qp_opts = {
            'max_iter': 10,
            'error_on_fail': False,
            'print_header': False,
            'print_iter': False
        }
        self.sol_options_sqp = {
            'max_iter': 3,
            'qpsol': 'qrqp',
            'convexify_margin': 1e-5,
            'print_header': False,
            'print_time': False,
            'print_iteration': False,
            'qpsol_options': qp_opts
        }

        # Options for IPOPT Solver
        # -> IPOPT
        self.sol_options_ipopt = {
            'ipopt.print_level': 0,
            'ipopt.warm_start_bound_push': 1e-4,
            'ipopt.warm_start_bound_frac': 1e-4,
            'ipopt.warm_start_slack_bound_frac': 1e-4,
            'ipopt.warm_start_slack_bound_push': 1e-4,
            'ipopt.warm_start_mult_bound_push': 1e-4,
            'print_time': False,
            'verbose': False,
        }

        return True
    
    def set_solver_dictionaries(self, nlp):

        self.solver_dict = {
            'sqpmethod': ca.nlpsol('mpc_solver', 'sqpmethod', nlp,
                                   self.sol_options_sqp),
            'ipopt': ca.nlpsol('mpc_solver', 'ipopt', nlp,
                               self.sol_options_ipopt)
        }
        return
    
    
    def set_cost_functions(self):
        """
        Helper function to setup the cost functions.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx, self.Nx)
        P = ca.MX.sym('P', self.Nx, self.Nx)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nx)
        xr = ca.MX.sym('xr', self.Nx)
        u = ca.MX.sym('u', self.Nu)

        # Calculate errors
        err = x - xr
        err_vec = ca.vertcat(*[err])

        # Calculate running cost
        ln = err_vec.T @ Q @ err_vec + u.T @ R @ u
        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln], self.fun_options)

        # Calculate terminal cost
        V = err_vec.T @ P @ err_vec 
        self.terminal_cost = ca.Function('V', [x, xr, P], [V], self.fun_options)
        return
    
    def solve_mpc(self, x0, u0=None):
        """
        Solve the MPC problem
        - param x0: state (type x0: ca.DM)
        = return: predicted states and control inputs (type: ca.DM ca.DM vectors)
        """

        param = x0
        args = dict(x0 = self.optvar_init,
                    lbx = self.optvar_lb,
                    ubx = self.optvar_ub,
                    lbg = self.con_lb,
                    ubg = self.con_ub,
                    p = param)


        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        

        # Solve NLP 
        solve_time = -time.time()
        sol = self.solver(**args)
        solve_time += time.time()
        optvar = self.opt_var(sol['x'])
        self.solve_time = solve_time

        status = None
        if self.solver_type == "ipopt":
            status = self.solver.stats()['return_status']
        elif self.solver_type == "sqpmethod":
            status = self.solver.stats()['success']

        print('Solver status: ', status)
        print('MPC cost: ', sol['f'])

        return optvar['x'], optvar['u']

if __name__ == "__main__":
    controller = NonlinearMPC()