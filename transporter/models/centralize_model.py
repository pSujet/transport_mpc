import casadi as ca
import numpy as np
from transporter.utils.function import *

class CentralizedModel:
    '''
    Centralized model of the load transportation system
    model all transporters and load as a single system
    (1)--(L)--(2)
    '''
    # Parameters of the system
    m1 = 1                                                          # mass robot 1 [kg]
    m2 = 1                                                          # mass robot 2 [kg]
    mL = 2                                                          # mass Load [kg]
    J1 = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia robot 1 [kgm^2]
    J2 = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia robot 2 [kgm^2]
    JL = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia Load [kgm^2]
    r1 = ca.DM([[0.5],[0],[0]])                                     # vector from CM of robot 1 to anchor point 
    r2 = ca.DM([[-0.5],[0],[0]])                                    # vector from CM of robot 2 to anchor point 
    R1 = ca.DM([[-1],[0],[0]])                                      # vector from CM of Load to anchor point 1
    R2 = ca.DM([[1],[0],[0]])                                       # vector from CM of Load to anchor point 2
    l1 = 3                                                          # cable 1 length [m]
    l2 = 3                                                          # cable 2 length [m]
    dt = 0.1                                                       # sampling rate [s], 50 Hz 
    D1 = ca.DM([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])         # thrust direction matrix robot 1
    L1 = ca.DM([[0,0,0,0,1,-1],[1,-1,0,0,0,0],[0,0,-1,1,0,0]])*0.5  # thrust arm matrix robot 1
    D2 = ca.DM([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])         # thrust direction matrix robot 2
    L2 = ca.DM([[0,0,0,0,1,-1],[1,-1,0,0,0,0],[0,0,-1,1,0,0]])*0.5  # thrust arm matrix robot 2

    def __init__(self) -> None:
        self.n = 39
        self.m = 12
        # -- setup dynamics model for simulation
        self.central_dynamics()
        # -- setup dynamics model for controller
        self.control_dynamics()
        self.scale = 10
        # -- setup integrator
        self.rk4_integrator()

        # -- linearized discretized 
        # self.linearized_discretized()
        # self.linearized_model()
        # self.rk4_integrator_linear(self.lindynamics)

        # -- discretized linearized 
        self.linearized()        
        

    def central_dynamics(self):
        """
            CENTRALIZED DYNAMICS
                (1)--(L)--(2)
            
            -- states of the system --
            x1 = [p1x,p1y,p1z,v1x,v1y,v1z,q11,q12,q13,q14,w1x,w1y,w1z]
            x2 = [p2x,p2y,p2z,v2x,v2y,v2z,q21,q22,q23,q24,w2x,w2y,w2z]
            xL = [pLx,pLy,pLz,vLx,vLy,vLz,qL1,qL2,qL3,qL4,wLx,wLy,wLz]
            u = [F1x,F1y,F1z,T1x,T1y,T1z,F2x,F2y,F2z,T2x,T2y,T2z]
            
            x = [x1,x2,xL]
            u = [F1,T1,F2,T2]
        """
        x = ca.MX.sym('x',self.n,1)
        u = ca.MX.sym('u',self.m,1)

        # == Extract variable for simplicity
        # -- states robot 1
        p1 = x[0:3]
        v1 = x[3:6]
        q1 = x[6:10]
        w1 = x[10:13]

        # -- states robot 2
        p2 = x[13:16]
        v2 = x[16:19]
        q2 = x[19:23]
        w2 = x[23:26]

        # -- states Load
        pL = x[26:29]
        vL = x[29:32]
        qL = x[32:36]
        wL = x[36:39]

        # -- inputs --
        # F1 = u[0:3]                 # input force in body frame (robot 1)
        # T1 = u[3:6]                 # input torque in body frame (robot 1)
        # F2 = u[6:9]                 # input force in body frame (robot 2)
        # T2 = u[9:12]                # input force in body frame (robot 2)
        F1 = self.D1 @ u[0:6]              # input force in body frame (robot 1)
        T1 = self.L1 @ u[0:6]              # input torque in body frame (robot 1)
        F2 = self.D2 @ u[6:12]             # input force in body frame (robot 2)
        T2 = self.L2 @ u[6:12]             # input force in body frame (robot 2)

        # == Extract parameters for simplicity
        m1 = self.m1
        m2 = self.m2
        mL = self.mL
        J1 = self.J1
        J2 = self.J2
        JL = self.JL
        r1 = self.r1
        r2 = self.r2
        R1 = self.R1
        R2 = self.R2
        l1 = self.l1
        l2 = self.l2
        dt = self.dt

        #=== Kinematics of the system ===#
        # -- Direction Cosine Matrix
        Ld1 = dcm_e2b(q1)
        Ld2 = dcm_e2b(q2)
        LdL = dcm_e2b(qL)

        # -- Kinematics
        p1_prime = p1 + Ld1.T @ r1
        v1_prime = v1 + Ld1.T @ (skew(w1) @ r1)
        s1 = pL + LdL.T @ R1
        vs1 = vL + LdL.T @ (skew(wL) @ R1)
        vdiff1_norm = (v1_prime - vs1).T @ (v1_prime - vs1)

        p2_prime = p2 + Ld2.T @ r2
        v2_prime = v2 + Ld2.T @ (skew(w2) @ r2)
        s2 = pL + LdL.T @ R2
        vs2 = vL + LdL.T @ (skew(wL) @ R2)
        vdiff2_norm = (v2_prime - vs2).T @ (v2_prime - vs2)

        # -- compute b unit vector for linearization
        b =  (s1 - s2)/ np.sqrt((s1 - s2).T @ (s1 - s2))


        # -- Compute tension (Ts)
        n1 = (p1_prime-s1)/l1
        n2 = (p2_prime-s2)/l2

        # -- cable 1
        mu1 = Ld1.T @ (F1/m1 - skew(r1) @ ca.inv(J1) @ (-skew(w1)@J1@w1 + T1) + skew(w1)@skew(w1)@r1) - LdL.T@(-skew(R1)@ca.inv(JL)@(-skew(wL)@JL@wL)+skew(wL)@skew(wL)@R1)
        phi1 = (m1 + mL)/(m1*mL)*ca.DM.eye(3) - Ld1.T@skew(r1)@ca.inv(J1)@skew(r1)@Ld1 - LdL.T@skew(R1)@ca.inv(JL)@skew(R1)@LdL
        sigma1 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R1)@ca.inv(JL)@skew(R2)@LdL
        zeta1 = mu1.T @ n1 + vdiff1_norm/l1
        alpha1 = (sigma1@n2).T@n1
        beta1 = (phi1@n1).T@n1

        # -- cable 2
        mu2 = Ld2.T @ (F2/m2 - skew(r2) @ ca.inv(J2) @ (-skew(w2)@J2@w2 + T2) + skew(w2)@skew(w2)@r2) - LdL.T@(-skew(R2)@ca.inv(JL)@(-skew(wL)@JL@wL)+skew(wL)@skew(wL)@R2)
        phi2 = (m2 + mL)/(m2*mL)*ca.DM.eye(3) - Ld2.T@skew(r2)@ca.inv(J2)@skew(r2)@Ld2 - LdL.T@skew(R2)@ca.inv(JL)@skew(R2)@LdL
        sigma2 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R2)@ca.inv(JL)@skew(R1)@LdL
        zeta2 = mu2.T @ n2 + vdiff2_norm/l2
        alpha2 = ((sigma2@n1).T@n2)
        beta2 = ((phi2@n2).T@n2)

        # -- Tension force [N]
        Ts1 = (zeta1*beta2-zeta2*alpha1)/((beta1*beta2-alpha1*alpha2))
        Ts2 = (zeta2*beta1-zeta1*alpha2)/((beta1*beta2-alpha1*alpha2))

        # -- Tension vector
        Tv1 = Ts1*n1
        Tv2 = Ts2*n2

        # -- Add sigmoid function       
        # if np.sqrt(n1.T @ n1) <= 1:
        #     Tv1 = Tv1 * 0
        # if np.sqrt(n2.T @ n2) <= 1:
        #     Tv2 = Tv2 * 0
        # Tv1 = Tv1 * 1/(1+np.exp(100010-100000*np.sqrt(n1.T @ n1)))
        # Tv2 = Tv2 * 1/(1+np.exp(100010-100000*np.sqrt(n2.T @ n2)))

        #=== Dynamics of the system ===#
        # -- Equation of motion ODE

        # -- robot 1
        p1dot = v1
        v1dot = 1/m1*(Ld1.T@F1-Tv1)
        q1dot = 1/2 * omega_oper(w1) @ q1
        w1dot = ca.inv(J1)@(-skew(w1)@J1@w1 + T1 - skew(r1)@(Ld1@Tv1))

        # -- robot 2
        p2dot = v2
        v2dot = 1/m2*(Ld2.T@F2-Tv2)
        q2dot = 1/2 * omega_oper(w2) @ q2
        w2dot = ca.inv(J2)@(-skew(w2)@J2@w2 + T2 - skew(r2)@(Ld2@Tv2))

        # -- Load
        pLdot = vL
        vLdot = 1/mL*(Tv1 + Tv2)
        qLdot = 1/2 * omega_oper(wL) @ qL
        wLdot = ca.inv(JL)@(-skew(wL)@JL@wL + skew(R1)@(LdL@Tv1) + skew(R2)@(LdL@Tv2))

        dxdt = ca.vertcat(p1dot,v1dot,q1dot,w1dot,p2dot,v2dot,q2dot,w2dot,pLdot,vLdot,qLdot,wLdot)

        # --  compute angle diff from equilibrium for cost function
        angle_diff1 = b.T @ n1
        angle_diff2 = -b.T @ n2


        # set function
        self.n1F = ca.Function('n1F',[x,u],[n1], ['xk', 'uk'], ['n1'])
        self.n2F = ca.Function('n2F',[x,u],[n2], ['xk', 'uk'], ['n2'])
        self.bF = ca.Function('bF',[x,u],[b], ['xk', 'uk'], ['b'])
        self.tension1 = ca.Function('tension1',[x,u],[Tv1], ['xk', 'uk'], ['Tv1'])
        self.tension2 = ca.Function('tension2',[x,u],[Tv2], ['xk', 'uk'], ['Tv2'])
        self.dynamics = ca.Function('dynamics', [x, u], [dxdt], ['x', 'u'], ['dxdt'])
        self.angle_diff1 = ca.Function('angle_diff1',[x,u],[angle_diff1], ['xk', 'uk'], ['angle_diff1'])
        self.angle_diff2 = ca.Function('angle_diff2',[x,u],[angle_diff2], ['xk', 'uk'], ['angle_diff2'])

    def control_dynamics(self):
        """
            CENTRALIZED DYNAMICS
                (1)--(L)--(2)
            
            -- states of the system --
            x1 = [p1x,p1y,p1z,v1x,v1y,v1z,q11,q12,q13,q14,w1x,w1y,w1z]
            x2 = [p2x,p2y,p2z,v2x,v2y,v2z,q21,q22,q23,q24,w2x,w2y,w2z]
            xL = [pLx,pLy,pLz,vLx,vLy,vLz,qL1,qL2,qL3,qL4,wLx,wLy,wLz]
            u = [F1x,F1y,F1z,T1x,T1y,T1z,F2x,F2y,F2z,T2x,T2y,T2z]
            
            x = [x1,x2,xL]
            u = [F1,T1,F2,T2]
        """
        x = ca.MX.sym('x',self.n,1)
        u = ca.MX.sym('u',self.m,1)
        lg_multiplier = ca.MX.sym('lambda',2,1)
        lambda1 = lg_multiplier[0]
        lambda2 = lg_multiplier[1]


        # == Extract variable for simplicity
        # -- states robot 1
        p1 = x[0:3]
        v1 = x[3:6]
        q1 = x[6:10]
        w1 = x[10:13]

        # -- states robot 2
        p2 = x[13:16]
        v2 = x[16:19]
        q2 = x[19:23]
        w2 = x[23:26]

        # -- states Load
        pL = x[26:29]
        vL = x[29:32]
        qL = x[32:36]
        wL = x[36:39]

        # -- inputs --
        # F1 = u[0:3]                 # input force in body frame (robot 1)
        # T1 = u[3:6]                 # input torque in body frame (robot 1)
        # F2 = u[6:9]                 # input force in body frame (robot 2)
        # T2 = u[9:12]                # input force in body frame (robot 2)
        F1 = self.D1 @ u[0:6]              # input force in body frame (robot 1)
        T1 = self.L1 @ u[0:6]              # input torque in body frame (robot 1)
        F2 = self.D2 @ u[6:12]             # input force in body frame (robot 2)
        T2 = self.L2 @ u[6:12]             # input force in body frame (robot 2)

        # == Extract parameters for simplicity
        m1 = self.m1
        m2 = self.m2
        mL = self.mL
        J1 = self.J1
        J2 = self.J2
        JL = self.JL
        r1 = self.r1
        r2 = self.r2
        R1 = self.R1
        R2 = self.R2
        l1 = self.l1
        l2 = self.l2
        dt = self.dt

        #=== Kinematics of the system ===#
        # -- Direction Cosine Matrix
        Ld1 = dcm_e2b(q1)
        Ld2 = dcm_e2b(q2)
        LdL = dcm_e2b(qL)

        # -- Kinematics
        p1_prime = p1 + Ld1.T @ r1
        v1_prime = v1 + Ld1.T @ (skew(w1) @ r1)
        s1 = pL + LdL.T @ R1
        vs1 = vL + LdL.T @ (skew(wL) @ R1)
        vdiff1_norm = (v1_prime - vs1).T @ (v1_prime - vs1)

        p2_prime = p2 + Ld2.T @ r2
        v2_prime = v2 + Ld2.T @ (skew(w2) @ r2)
        s2 = pL + LdL.T @ R2
        vs2 = vL + LdL.T @ (skew(wL) @ R2)
        vdiff2_norm = (v2_prime - vs2).T @ (v2_prime - vs2)

        # Interaction force with Lagrange Multiplier
        Tv1 = lambda1 * (p1_prime-s1)
        Tv2 = lambda2 * (p2_prime-s2)

        # -- compute b unit vector for linearization
        b =  (s1 - s2)/ np.sqrt((s1 - s2).T @ (s1 - s2))

        # -- Compute cable unit vector direction
        n1 = (p1_prime-s1)/l1
        n2 = (p2_prime-s2)/l2

        #=== Dynamics of the system ===#
        # -- Equation of motion ODE

        # -- robot 1
        p1dot = v1
        v1dot = 1/m1*(Ld1.T@F1-Tv1)
        q1dot = 1/2 * omega_oper(w1) @ q1
        w1dot = ca.inv(J1)@(-skew(w1)@J1@w1 + T1 - skew(r1)@(Ld1@Tv1))

        # -- robot 2
        p2dot = v2
        v2dot = 1/m2*(Ld2.T@F2-Tv2)
        q2dot = 1/2 * omega_oper(w2) @ q2
        w2dot = ca.inv(J2)@(-skew(w2)@J2@w2 + T2 - skew(r2)@(Ld2@Tv2))

        # -- Load
        pLdot = vL
        vLdot = 1/mL*(Tv1 + Tv2)
        qLdot = 1/2 * omega_oper(wL) @ qL
        wLdot = ca.inv(JL)@(-skew(wL)@JL@wL + skew(R1)@(LdL@Tv1) + skew(R2)@(LdL@Tv2))

        # -- Holonomic constraint
        a1_prime = v1dot + Ld1.T @ (-skew(r1)@w1dot + skew(w1)@skew(w1)@r1)
        as1 = vLdot + LdL.T @ (-skew(R1)@wLdot + skew(wL)@skew(wL)@R1)
        a2_prime = v2dot + Ld2.T @ (-skew(r2)@w2dot + skew(w2)@skew(w2)@r2)
        as2 = vLdot + LdL.T @ (-skew(R2)@wLdot + skew(wL)@skew(wL)@R2)

        psi1ddot = (p1_prime - s1).T @ (a1_prime - as1) + vdiff1_norm
        psi2ddot = (p2_prime - s2).T @ (a2_prime - as2) + vdiff2_norm
        self.psi1ddotF = ca.Function('psi1ddotF',[x,u,lg_multiplier],[psi1ddot], ['xk', 'uk','lg_multiplier'], ['psi1ddot'])
        self.psi2ddotF = ca.Function('psi2ddotF',[x,u,lg_multiplier],[psi2ddot], ['xk', 'uk','lg_multiplier'], ['psi2ddot'])

        dxdt = ca.vertcat(p1dot,v1dot,q1dot,w1dot,p2dot,v2dot,q2dot,w2dot,pLdot,vLdot,qLdot,wLdot)

        self.ctrl_dynamics = ca.Function('ctrl_dynamics', [x, u, lg_multiplier], [dxdt], ['x', 'u', 'lg_multiplier'], ['dxdt'])

    def rk4_explicit(self,x0,u, dynamics,scale = 10):
        '''
        Explicit Runge-Kutta 4th Order discretization.
        '''
        x = x0
        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt/scale / 2 * k1, u)
        k3 = dynamics(x + self.dt/scale / 2 * k2, u)
        k4 = dynamics(x + self.dt/scale * k3, u)
        xf = x0 + self.dt /scale/ 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xf

    def rk4_integrator(self):
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)
        lg_multiplier = ca.MX.sym('lambda',2,1)

        xf = self.rk4_explicit(x0,u,self.dynamics,scale = 1)
        self.rk4 = ca.Function('rk4', [x0, u], [xf], ['xk', 'uk'], ['xf'])

        # -- use smaller dt for physical simulation
        xfsim = self.rk4_explicit(x0,u,self.dynamics,scale = 10)
        self.rk4sim = ca.Function('rk4sim', [x0, u], [xfsim], ['xk', 'uk'], ['xfsim'])

        # -- controller dynamics
        x = x0
        k1 = self.ctrl_dynamics(x, u, lg_multiplier)
        k2 = self.ctrl_dynamics(x + self.dt / 2 * k1, u, lg_multiplier)
        k3 = self.ctrl_dynamics(x + self.dt / 2 * k2, u, lg_multiplier)
        k4 = self.ctrl_dynamics(x + self.dt * k3, u, lg_multiplier)
        xfctrl = x0 + self.dt/ 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.rk4ctrl = ca.Function('rk4ctrl', [x0, u, lg_multiplier], [xfctrl], ['xk', 'uk','lg_multiplier'], ['xfctrl'])

    def get_equi_pos(self,pL,qL):
        return get_equi(pL,qL,self.R1,self.R2,self.l1,self.l2,self.r1,self.r2)
    
    def linearized(self):
        x = ca.MX.sym('x', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        # Create functions to get linearized matices evaluated at any points
        self.evalAd = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(self.rk4(x,u), x)])
        self.evalBd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(self.rk4(x,u), u)])

        # linearization point around 
        # (1)--(L)--(2)
        pL = ca.DM([[0],[0],[0]])
        qL = ca.DM([[0],[0],[0],[1]])
        qL = qL/np.linalg.norm(qL)
        # -- state robot depends on load at equilibrium
        p1,p2,q1,q2 = self.get_equi_pos(pL,qL)
        # -- all derivatives is zero
        v1 = ca.DM([[0],[0],[0]])
        w1 = ca.DM([[0],[0],[0]])
        w2 = ca.DM([[0],[0],[0]])
        vL = ca.DM([[0],[0],[0]])
        v2 = ca.DM([[0],[0],[0]])
        wL = ca.DM([[0],[0],[0]])
        x_bar = np.array(ca.vertcat(p1,v1,q1,w1,p2,v2,q2,w2,pL,vL,qL,wL))
        # u_bar = np.zeros((12, 1))
        u_bar = np.array([[-1],[-1],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0]])*0.1
        # u_bar = np.array([-1,-1,0,0,0,0,1,1,0,0,0,0])*0.1

        # get linearized matrix Ad and Bd
        # self.Ad = np.asarray(self.evalAd(x_bar, u_bar))
        # self.Bd = np.asarray(self.evalBd(x_bar, u_bar))
        self.Ad = ca.DM(self.evalAd(x_bar, u_bar))
        self.Bd = ca.DM(self.evalBd(x_bar, u_bar))
    
    def linearized_discretized(self):
        x_bar = ca.MX.sym('x', self.n, 1)
        u_bar = ca.MX.sym('u', self.m, 1)

        # Create functions to get linearized matices evaluated at any points
        self.evalAd = ca.Function('jac_x_Ad', [x_bar, u_bar], [ca.jacobian(self.dynamics(x_bar,u_bar), x_bar)])
        self.evalBd = ca.Function('jac_u_Bd', [x_bar, u_bar], [ca.jacobian(self.dynamics(x_bar,u_bar), u_bar)])

    def linearized_model(self):
        x = ca.MX.sym('x', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)
        A = ca.MX.sym('A',self.n,self.n)
        B = ca.MX.sym('B',self.n,self.m)
        deltaxdeltat = A @ x + B @ u
        self.lindynamics = ca.Function('lindynamics', [x, u, A, B], [deltaxdeltat], ['x', 'u','A','B'], ['deltaxdeltat'])
    
    def rk4_integrator_linear(self, dynamics):
        '''
        Explicit Runge-Kutta 4th Order discretization.
        '''
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)
        A = ca.MX.sym('A',self.n,self.n)
        B = ca.MX.sym('B',self.n,self.m)
        x = x0
        k1 = dynamics(x, u, A, B)
        k2 = dynamics(x + self.dt / 2 * k1, u, A, B)
        k3 = dynamics(x + self.dt / 2 * k2, u, A, B)
        k4 = dynamics(x + self.dt * k3, u, A, B)
        xf = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.rk4_lin = ca.Function('rk4', [x0, u, A, B], [xf], ['xk', 'uk','A','B'], ['xf'])













