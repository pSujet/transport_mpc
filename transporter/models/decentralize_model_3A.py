import casadi as ca
import numpy as np
from transporter.utils.function import *

class DecentralizedModel3A:
    '''
    Centralized model of the load transportation system
    model all transporters and load as a single system
    (1)--(L)--(2)
    '''
    # Parameters of the system
    m1 = 1                                                          # mass robot 1 [kg]
    m2 = 1                                                          # mass robot 2 [kg]
    m3 = 1                                                          # mass robot 3 [kg]
    mL = 2                                                          # mass Load [kg]
    J1 = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia robot 1 [kgm^2]
    J2 = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia robot 2 [kgm^2]
    J3 = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia robot 3 [kgm^2]
    JL = ca.DM([[1,0,0],[0,1,0],[0,0,1]])                           # moment of inertia Load [kgm^2]
    r1 = 0.5*ca.DM([[0],[-1],[0]])                                  # vector from CM of robot 1 to anchor point 
    r2 = 0.5*ca.DM([[np.cos(np.pi/6)],[np.sin(np.pi/6)],[0]])       # vector from CM of robot 2 to anchor point 
    r3 = 0.5*ca.DM([[-np.cos(np.pi/6)],[np.sin(np.pi/6)],[0]])      # vector from CM of robot 2 to anchor point 
    R1 = ca.DM([[0],[1],[0]])                                       # vector from CM of Load to anchor point 1
    R2 = ca.DM([[-np.sin(np.pi/3)],[-np.cos(np.pi/3)],[0]])         # vector from CM of Load to anchor point 2
    R3 = ca.DM([[np.sin(np.pi/3)],[-np.cos(np.pi/3)],[0]])          # vector from CM of Load to anchor point 3
    l1 = 3                                                          # cable 1 length [m]
    l2 = 3                                                          # cable 2 length [m]
    l3 = 3                                                          # cable 3 length [m]
    dt = 0.1                                                        # sampling rate [s], 50 Hz 
    D1 = ca.DM([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])         # thrust direction matrix robot 1
    L1 = ca.DM([[0,0,0,0,1,-1],[1,-1,0,0,0,0],[0,0,-1,1,0,0]])*0.5  # thrust arm matrix robot 1
    D2 = ca.DM([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])         # thrust direction matrix robot 2
    L2 = ca.DM([[0,0,0,0,1,-1],[1,-1,0,0,0,0],[0,0,-1,1,0,0]])*0.5  # thrust arm matrix robot 2
    D3 = ca.DM([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])         # thrust direction matrix robot 3
    L3 = ca.DM([[0,0,0,0,1,-1],[1,-1,0,0,0,0],[0,0,-1,1,0,0]])*0.5  # thrust arm matrix robot 3

    def __init__(self,rdc,Rdc1,Rdc2,Rdc3) -> None:
        self.n = 52
        self.m = 18
        self.Nx = 26
        self.Nu = 6
        self.rdc = rdc                                              # vector from CM of robot to anchor point 
        self.Rdc1 = Rdc1                                            # vector from CM of load to anchor point
        self.Rdc2 = Rdc2                                            # vector from CM of load to anchor point
        self.Rdc3 = Rdc3                                            # vector from CM of load to anchor point
        # -- setup dynamics model for simulation
        self.central_dynamics()
        # -- setup dynamics model for controller
        self.control_dynamics()
        self.scale = 10
        # -- setup integrator
        self.rk4_integrator()

        # -- discretized linearized 
        # self.linearized()        
        

    def central_dynamics(self):
        """
            CENTRALIZED DYNAMICS
                     (1)
                      |
                      |
                     (L)
                    /   \
                  /       \ 
                (2)        (3)
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
        
        # -- states robot 3
        p3 = x[26:29]
        v3 = x[29:32]
        q3 = x[32:36]
        w3 = x[36:39]

        # -- states Load
        pL = x[39:42]
        vL = x[42:45]
        qL = x[45:49]
        wL = x[49:52]

        # -- inputs --
        F1 = self.D1 @ u[0:6]              # input force in body frame (robot 1)
        T1 = self.L1 @ u[0:6]              # input torque in body frame (robot 1)
        F2 = self.D2 @ u[6:12]             # input force in body frame (robot 2)
        T2 = self.L2 @ u[6:12]             # input force in body frame (robot 2)
        F3 = self.D3 @ u[12:18]             # input force in body frame (robot 2)
        T3 = self.L3 @ u[12:18]             # input force in body frame (robot 2)

        # == Extract parameters for simplicity
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        mL = self.mL
        J1 = self.J1
        J2 = self.J2
        J3 = self.J3
        JL = self.JL
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3
        R1 = self.R1
        R2 = self.R2
        R3 = self.R3
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        dt = self.dt

        #=== Kinematics of the system ===#
        # -- Direction Cosine Matrix
        Ld1 = dcm_e2b(q1)
        Ld2 = dcm_e2b(q2)
        Ld3 = dcm_e2b(q3)
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

        p3_prime = p3 + Ld3.T @ r3
        v3_prime = v3 + Ld3.T @ (skew(w3) @ r3)
        s3 = pL + LdL.T @ R3
        vs3 = vL + LdL.T @ (skew(wL) @ R3)
        vdiff3_norm = (v3_prime - vs3).T @ (v3_prime - vs3)

        # -- compute b unit vector for linearization
        b =  (s1 - s2)/ np.sqrt((s1 - s2).T @ (s1 - s2))

        # -- Compute tension (Ts)
        n1 = (p1_prime-s1)/l1
        n2 = (p2_prime-s2)/l2
        n3 = (p3_prime-s3)/l3

        # -- cable 1
        mu1 = Ld1.T @ (F1/m1 - skew(r1) @ ca.inv(J1) @ (-skew(w1)@J1@w1 + T1) + skew(w1)@skew(w1)@r1) - LdL.T@(-skew(R1)@ca.inv(JL)@(-skew(wL)@JL@wL)+skew(wL)@skew(wL)@R1)
        phi1 = (m1 + mL)/(m1*mL)*ca.DM.eye(3) - Ld1.T@skew(r1)@ca.inv(J1)@skew(r1)@Ld1 - LdL.T@skew(R1)@ca.inv(JL)@skew(R1)@LdL
        sigma12 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R1)@ca.inv(JL)@skew(R2)@LdL
        sigma13 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R1)@ca.inv(JL)@skew(R3)@LdL
        zeta1 = mu1.T @ n1 + vdiff1_norm/l1
        alpha12 = (sigma12@n2).T@n1
        alpha13 = (sigma13@n3).T@n1
        beta1 = (phi1@n1).T@n1

        # -- cable 2
        mu2 = Ld2.T @ (F2/m2 - skew(r2) @ ca.inv(J2) @ (-skew(w2)@J2@w2 + T2) + skew(w2)@skew(w2)@r2) - LdL.T@(-skew(R2)@ca.inv(JL)@(-skew(wL)@JL@wL)+skew(wL)@skew(wL)@R2)
        phi2 = (m2 + mL)/(m2*mL)*ca.DM.eye(3) - Ld2.T@skew(r2)@ca.inv(J2)@skew(r2)@Ld2 - LdL.T@skew(R2)@ca.inv(JL)@skew(R2)@LdL
        sigma21 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R2)@ca.inv(JL)@skew(R1)@LdL
        sigma23 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R2)@ca.inv(JL)@skew(R3)@LdL
        zeta2 = mu2.T @ n2 + vdiff2_norm/l2
        alpha21 = ((sigma21@n1).T@n2)
        alpha23 = ((sigma23@n3).T@n2)
        beta2 = ((phi2@n2).T@n2)

        # -- cable 3
        mu3 = Ld3.T @ (F3/m3 - skew(r3) @ ca.inv(J3) @ (-skew(w3)@J3@w3 + T3) + skew(w3)@skew(w3)@r3) - LdL.T@(-skew(R3)@ca.inv(JL)@(-skew(wL)@JL@wL)+skew(wL)@skew(wL)@R3)
        phi3 = (m3 + mL)/(m3*mL)*ca.DM.eye(3) - Ld3.T@skew(r3)@ca.inv(J3)@skew(r3)@Ld3 - LdL.T@skew(R3)@ca.inv(JL)@skew(R3)@LdL
        sigma31 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R3)@ca.inv(JL)@skew(R1)@LdL
        sigma32 = 1/mL*ca.DM.eye(3) - LdL.T@skew(R3)@ca.inv(JL)@skew(R2)@LdL
        zeta3 = mu3.T @ n3 + vdiff3_norm/l3
        alpha31 = ((sigma31@n1).T@n3)
        alpha32 = ((sigma32@n2).T@n3)
        beta3 = ((phi3@n3).T@n3)

        # -- Tension force [N]
        # -- Matrix way
        # Ax = b
        # Amat = ca.vertcat(ca.horzcat(beta1,alpha12,alpha13),
        #                   ca.horzcat(alpha21,beta2,alpha23),
        #                   ca.horzcat(alpha31,alpha32,beta3))
        # bmat = ca.vertcat(zeta1,zeta2,zeta3)
        # Tmat = ca.inv(Amat)*bmat
        # Ts1 = Tmat[0,1]
        # Ts2 = Tmat[1,1]
        # Ts3 = Tmat[2,1]
        # -- explicit due to MX to SX
        invdetA = 1/(beta1*beta2*beta3 + alpha12*alpha23*alpha31 + alpha13*alpha21*alpha32 
                     - alpha13*beta2*alpha31 - alpha12*alpha21*beta3 - beta1*alpha23*alpha32)
        Ts1 = invdetA*((beta2*beta3 - alpha23*alpha32)*zeta1 - (alpha12*beta3 - alpha13*alpha32)*zeta2 + (alpha12*alpha23 - alpha13*beta2)*zeta3)
        Ts2 = invdetA*(-(alpha21*beta3 - alpha23*alpha31)*zeta1 + (beta1*beta3 - alpha13*alpha31)*zeta2 - (beta1*alpha23 - alpha13*alpha21)*zeta3)
        Ts3 = invdetA*((alpha21*alpha32 - beta2*alpha31)*zeta1 - (beta1*alpha32 - alpha12*alpha31)*zeta2 + (beta1*beta2 - alpha12*alpha21)*zeta3)

        # -- Tension vector
        Tv1 = Ts1*n1
        Tv2 = Ts2*n2
        Tv3 = Ts3*n3

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

        # -- robot 3
        p3dot = v3
        v3dot = 1/m3*(Ld3.T@F3-Tv3)
        q3dot = 1/2 * omega_oper(w3) @ q3
        w3dot = ca.inv(J3)@(-skew(w3)@J3@w3 + T3 - skew(r3)@(Ld3@Tv3))

        # -- Load
        pLdot = vL
        vLdot = 1/mL*(Tv1 + Tv2 + Tv3)
        qLdot = 1/2 * omega_oper(wL) @ qL
        wLdot = ca.inv(JL)@(-skew(wL)@JL@wL + skew(R1)@(LdL@Tv1) + skew(R2)@(LdL@Tv2) + skew(R3)@(LdL@Tv3)) 

        dxdt = ca.vertcat(p1dot,v1dot,q1dot,w1dot,p2dot,v2dot,q2dot,w2dot,p3dot,v3dot,q3dot,w3dot,pLdot,vLdot,qLdot,wLdot)

        # set function
        self.n1F = ca.Function('n1F',[x,u],[n1], ['xk', 'uk'], ['n1'])
        self.n2F = ca.Function('n2F',[x,u],[n2], ['xk', 'uk'], ['n2'])
        self.n3F = ca.Function('n3F',[x,u],[n3], ['xk', 'uk'], ['n3'])
        self.bF = ca.Function('bF',[x,u],[b], ['xk', 'uk'], ['b'])
        self.tension1 = ca.Function('tension1',[x,u],[Tv1], ['xk', 'uk'], ['Tv1'])
        self.tension2 = ca.Function('tension2',[x,u],[Tv2], ['xk', 'uk'], ['Tv2'])
        self.tension3 = ca.Function('tension2',[x,u],[Tv3], ['xk', 'uk'], ['Tv3'])
        self.dynamics = ca.Function('dynamics', [x, u], [dxdt], ['x', 'u'], ['dxdt'])


    def control_dynamics(self):
        """
            DE CENTRALIZED DYNAMICS
                     (1)
                      |
                      |
                     (L)
                    /   \
                  /       \ 
                (T)        (T)
        """  
        x = ca.MX.sym('x',self.Nx,1)
        u = ca.MX.sym('u',self.Nu,1)
        lg_multiplier = ca.MX.sym('lambda',1,1)
        lambda1 = lg_multiplier[0]

        # == Extract variable for simplicity
        # -- states robot 1
        p1 = x[0:3]
        v1 = x[3:6]
        q1 = x[6:10]
        w1 = x[10:13]

        # -- states robot 2
        pL = x[13:16]
        vL = x[16:19]
        qL = x[19:23]
        wL = x[23:26]
        
        # -- inputs --
        F1 = self.D1 @ u[0:6]              # input force in body frame (robot 1)
        T1 = self.L1 @ u[0:6]              # input torque in body frame (robot 1)

        # == Extract parameters for simplicity
        m1 = self.m1
        mL = self.mL/3
        J1 = self.J1
        JL = self.JL/3
        r1 = self.rdc
        R1 = self.Rdc1
        R2 = self.Rdc2
        R3 = self.Rdc3
        l1 = self.l1
        dt = self.dt

        #=== Kinematics of the system ===#
        # -- Direction Cosine Matrix
        Ld1 = dcm_e2b(q1)
        LdL = dcm_e2b(qL)

        # -- Kinematics
        p1_prime = p1 + Ld1.T @ r1
        v1_prime = v1 + Ld1.T @ (skew(w1) @ r1)
        s1 = pL + LdL.T @ R1
        vs1 = vL + LdL.T @ (skew(wL) @ R1)
        vdiff1_norm = (v1_prime - vs1).T @ (v1_prime - vs1)

        # -- Interaction force with Lagrange Multiplier
        Tv1 = lambda1 * (p1_prime-s1)/l1
        n1 = (p1_prime-s1)/l1

        LdL = dcm_e2b(qL)
        s1 = pL + LdL.T @ R1
        s2 = pL + LdL.T @ R2
        s3 = pL + LdL.T @ R3
        c = (s1+s2+s3)/3
        b1 = (s1 - c)/np.sqrt((s1 - c).T @ (s1 - c)) 
        b2 = (s2 - c)/np.sqrt((s2 - c).T @ (s2 - c)) 
        b3 = (s3 - c)/np.sqrt((s3 - c).T @ (s3 - c)) 

        virtual_tension = 1
        Tv2 = virtual_tension*b2
        Tv3 = virtual_tension*b3

        #=== Dynamics of the system ===#
        # -- Equation of motion ODE

        # -- robot 1
        p1dot = v1
        v1dot = 1/m1*(Ld1.T@F1-Tv1)
        q1dot = 1/2 * omega_oper(w1) @ q1
        w1dot = ca.inv(J1)@(-skew(w1)@J1@w1 + T1 - skew(r1)@(Ld1@Tv1))

        # -- Load
        pLdot = vL
        vLdot = 1/mL*(Tv1 + Tv2 + Tv3)
        qLdot = 1/2 * omega_oper(wL) @ qL
        wLdot = ca.inv(JL)@(-skew(wL)@JL@wL + skew(R1)@(LdL@Tv1) + skew(R2)@(LdL@Tv2) + skew(R3)@(LdL@Tv3)) 

        # -- Holonomic constraint
        a1_prime = v1dot + Ld1.T @ (-skew(r1)@w1dot + skew(w1)@skew(w1)@r1)
        as1 = vLdot + LdL.T @ (-skew(R1)@wLdot + skew(wL)@skew(wL)@R1)

        psi1ddot = (p1_prime - s1).T @ (a1_prime - as1) + vdiff1_norm

        self.psi1ddotF = ca.Function('psi1ddotF',[x,u,lg_multiplier],[psi1ddot], ['xk', 'uk','lg_multiplier'], ['psi1ddot'])

        dxdt = ca.vertcat(p1dot,v1dot,q1dot,w1dot,pLdot,vLdot,qLdot,wLdot)

        self.ctrl_dynamics = ca.Function('ctrl_dynamics', [x, u, lg_multiplier], [dxdt], ['x', 'u', 'lg_multiplier'], ['dxdt'])
        self.ctrln1F = ca.Function('n1F',[x,u],[n1], ['xk', 'uk'], ['n1'])
        self.ctrlb1F = ca.Function('b1F',[x,u],[n1], ['xk', 'uk'], ['b1'])

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
        lg_multiplier = ca.MX.sym('lambda',1,1)

        xf = self.rk4_explicit(x0,u,self.dynamics,scale = 1)
        self.rk4 = ca.Function('rk4', [x0, u], [xf], ['xk', 'uk'], ['xf'])

        # -- use smaller dt for physical simulation
        xfsim = self.rk4_explicit(x0,u,self.dynamics,scale = 10)
        self.rk4sim = ca.Function('rk4sim', [x0, u], [xfsim], ['xk', 'uk'], ['xfsim'])

        # -- controller dynamics
        x0 = ca.MX.sym('x0', self.Nx, 1)
        u = ca.MX.sym('u', self.Nu, 1)
        x = x0
        k1 = self.ctrl_dynamics(x, u, lg_multiplier)
        k2 = self.ctrl_dynamics(x + self.dt / 2 * k1, u, lg_multiplier)
        k3 = self.ctrl_dynamics(x + self.dt / 2 * k2, u, lg_multiplier)
        k4 = self.ctrl_dynamics(x + self.dt * k3, u, lg_multiplier)
        xfctrl = x0 + self.dt/ 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.rk4ctrl = ca.Function('rk4ctrl', [x0, u, lg_multiplier], [xfctrl], ['xk', 'uk','lg_multiplier'], ['xfctrl'])

        # -- Forwar Euler
        x0 = ca.MX.sym('x0', self.Nx, 1)
        u = ca.MX.sym('u', self.Nu, 1)
        xfeuler = x0 + self.dt * self.ctrl_dynamics(x0, u, lg_multiplier)
        self.feulctrl = ca.Function('feulctrl', [x0, u, lg_multiplier], [xfeuler], ['xk', 'uk','lg_multiplier'], ['xfctrl'])


    def get_equi_pos(self,pL,qL):
        return get_equi_3A(pL,qL,self.R1,self.R2,self.R3,self.l1,self.l2,self.l3,self.r1,self.r2,self.r3)
    
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













