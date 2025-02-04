import time

import jax.numpy as jnp
import matplotlib.pylab as plt
import torch
from casadi import *
from jax import random
from skopt.sampler import Hammersly
from torch import Tensor, autograd
from torch.nn import Sequential, Linear, ReLU
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from lipchitz_network import SequentialLipschitz


def ODE_RHS(x, u):
    # RHS of the aircraft dynamics
    xdot = [
        u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1], u[2] * np.cos(x[5]), u[2] * np.sin(x[5]),
        u[3]
    ]
    return np.array(xdot)


def dynamics_RK4(x, u, T, N):
    # RK4 integrator
    M = 4  # RK4 steps per interval
    DT = T / N / M
    X = np.array(x)
    U = np.array(u)
    for j in range(M):
        k1 = ODE_RHS(X, U)
        k2 = ODE_RHS(X + DT / 2 * k1, U)
        k3 = ODE_RHS(X + DT / 2 * k2, U)
        k4 = ODE_RHS(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return list(X)


def ZCBF(x):
    # The constructive CBF in [Squires et al]
    A1 = db(x) ** 2 + dc(x) ** 2 + 2 * (v / w) ** 2 - 2 * ((v / w) ** 2) * np.cos(x[2] - x[5])
    A2 = compute_A2(x)
    h = A1 - A2 - Ds ** 2
    return h


def num_grad(x):
    # Numerically compute the gradient of CBF h(x) w.r.t. x
    eps_grad = 1e-10
    dhdx = []
    for i in range(x.size):
        Eps = 0 * x
        Eps[i] = eps_grad
        dhdx.append((ZCBF(x + Eps) - ZCBF(x - Eps)) / (2 * eps_grad))
    return dhdx  # return a list


def generate_init_condition(R, num_points):
    # Generate initial conditions on a circle with radius R
    angle_vec = np.linspace(0, 2 * np.pi, num=num_points)
    X_start = []
    X_goal = []
    for angle in angle_vec:
        temp = R * np.exp(angle * 1j)
        x_start = [temp.real, temp.imag, np.pi + angle, -temp.real, -temp.imag, angle]
        if use_human_data:
            if temp.real >= 0:
                x_goal = [-2000, 0, np.pi, 2000, 0, 0]
            else:
                x_goal = [2000, 0, 0, -2000, 0, np.pi]
        else:
            if temp.real >= 0:
                x_goal = [-5, 0, np.pi, 5, 0, 0]
            else:
                x_goal = [5, 0, 0, -5, 0, np.pi]
        X_start.append(x_start)
        X_goal.append(x_goal)
    return np.array(X_start), np.array(X_goal)


def generate_training_data(X_start, X_goal, N_sim):
    # Generate expert demonstrations
    X_train = []
    U_train = []
    for i in range(X_start[:, 1].size):
        Xcl, Ucl, _ = safe_closed_loop(
            list(X_start[i, :]), list(X_goal[i, :]), N_sim, learning=0, plot=0, early_stop=1
        )
        X_train.append(Xcl[:-1, :])
        U_train.append(Ucl)
    X_train = np.vstack(X_train)
    U_train = np.vstack(U_train)
    return X_train, U_train


def get_ring_data(X, U, R_lb, R_ub, verbose):
    # Obtain the data within the ring of inner radius R_lb and outer radius R_ub
    condition = np.sqrt((X[:, 0] - X[:, 3]) ** 2 + (X[:, 1] - X[:, 4]) ** 2)

    mask = (condition >= R_lb) & (condition <= R_ub)

    X_out = np.array(X)
    U_out = np.array(U)

    X_out = X_out[mask]

    if U is not None:
        U_out = U_out[mask]

    if verbose:
        print(X_out.shape[0], 'points selected out from', X.shape[0], 'given points')
    return X_out, U_out


def safe_closed_loop(x_init, x_goal, N_sim, learning, plot, early_stop):
    # Closed-loop Simulations with CBF+MPC
    Xcl = []
    Ucl = []
    Xcl.append(np.array(x_init))
    xt = x_init
    sep_min = np.sqrt((x_init[0] - x_init[3]) ** 2 + (x_init[1] - x_init[4]) ** 2)
    count = 0

    hs = []
    dhdxs = []

    for t in range(N_sim):
        u_opt = CFTOC_NMPC(xt, x_goal, T, N)  # Solve the NMPC problem at the current time
        u_mod, f_mod, h, dhdx = CBF_QP(xt, u_opt, learning)  # Solve the CBF-QP problem at the current time

        hs.append(h)
        dhdxs.append(dhdx)

        if np.linalg.norm(np.array(u_opt) - np.array(u_mod)) <= 0.1:
            count += 1
        sep_now = np.sqrt((xt[0] - xt[3]) ** 2 + (xt[1] - xt[4]) ** 2)
        # (optional) disengage the CBF in the interior of the safe set
        if use_human_data:
            if sep_now >= 1.2 * Ds:
                u_mod = u_opt
        if sep_now < sep_min:
            sep_min = sep_now
        if early_stop and count >= 5:
            break
        xt = dynamics_RK4(xt, u_mod, T, N)  # Evolve the system
        Xcl.append(np.array(xt))
        Ucl.append(np.array(u_mod))
    Xcl = np.array(Xcl)
    Ucl = np.array(Ucl)
    if plot:
        tgrid = [ts * k for k in range(N_sim)]
        plt.figure(figsize=(10, 5))
        plt.clf()
        plt.plot(Xcl[:, 0], Xcl[:, 1], '-')
        plt.plot(Xcl[:, 3], Xcl[:, 4], '-')
        plt.xlabel('$p_x$')
        plt.ylabel('$p_y$')
        plt.legend(['agent a', 'agent b'])
        plt.xlim([-3, 3])
        plt.ylim([-2, 2])
        plt.axis('equal')
        plt.grid()
        # plt.savefig('closed_loop_plane.pdf', dpi=300)
        plt.figure()
        plt.clf()
        plt.step(tgrid, Ucl[:, 0], '-.')
        plt.step(tgrid, Ucl[:, 1], '-.')
        plt.step(tgrid, Ucl[:, 2], '-.')
        plt.step(tgrid, Ucl[:, 3], '-.')
        plt.xlabel('t')
        plt.legend(['u1', 'u2', 'u3', 'u4'])
        plt.grid()

        plt.figure()
        plt.clf()
        plt.step(tgrid, np.array(hs))
        plt.xlabel('t')
        plt.legend(['h'])
        plt.grid()

        plt.figure()
        plt.clf()
        plt.step(tgrid, np.array(dhdxs))
        plt.xlabel('t')
        plt.legend(['dx1', 'dy1', 'dt1', 'dx2', 'dy2', 'dt2'])
        plt.grid()

        plt.figure()
        plt.clf()

        tgrid = [ts * k for k in range(N_sim + 1)]
        distance = np.sqrt((Xcl[:, 0] - Xcl[:, 3]) ** 2 + (Xcl[:, 1] - Xcl[:, 4]) ** 2) - 0.5
        plt.plot(tgrid, distance)
        plt.xlabel('t')
        plt.legend(['contraint'])
        plt.grid()

        plt.show()
    return Xcl, Ucl, sep_min


def plot_closed_loop(Xcl, Ucl, timesteps):
    # Plot closed-loop state and input trajectories
    for t in timesteps:
        tgrid = [ts * k for k in range(t)]
        plt.figure()
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(Xcl[:t, 0], Xcl[:t, 1], '-')
        plt.plot(Xcl[:t, 3], Xcl[:t, 4], '-')
        plt.xlabel('x')
        plt.legend(['agent a', 'agent b'])
        plt.grid()
        plt.xlim([-3, 3])
        plt.ylim([-2, 2])
        plt.subplot(1, 2, 2)
        plt.step(tgrid, Ucl[:t, 0], '-.')
        plt.step(tgrid, Ucl[:t, 1], '-.')
        plt.step(tgrid, Ucl[:t, 2], '-.')
        plt.step(tgrid, Ucl[:t, 3], '-.')
        plt.xlabel('t')
        plt.legend(['u1', 'u2', 'u3', 'u4'])
        plt.grid()
        plt.show()


def CBF_QP(x, u_opt, learning):
    # Solve the CBF-QP problem
    x = np.array(x)  # Current states
    u1 = SX.sym('u1')  # Decision variables
    u2 = SX.sym('u2')
    u3 = SX.sym('u3')
    u4 = SX.sym('u4')
    u = vertcat(u1, u2, u3, u4)
    V = (u1 - u_opt[0]) ** 2 + (u2 - u_opt[1]) ** 2 + (u3 - u_opt[2]
                                                       ) ** 2 + (u4 - u_opt[3]) ** 2  # Objective function
    lbu = u_lb  # Variable bounds
    ubu = u_ub
    if learning:

        in_x = torch.from_numpy(x).type(torch.float)
        in_x.requires_grad = True
        hx = h_model(in_x)

        dhdx, = autograd.grad(hx, in_x,
                              grad_outputs=hx.data.new(hx.shape).fill_(1),
                              create_graph=True)

        hx = hx.detach().numpy()
        dhdx = dhdx.detach().numpy()
    else:
        hx = ZCBF(x)
        dhdx = num_grad(x)
    g = [dhdx[0] * (u1 * np.cos(x[2])) + dhdx[1] * (u1 * np.sin(x[2])) + dhdx[2] * u2 + \
         dhdx[3] * (u3 * np.cos(x[5])) + dhdx[4] * (u3 * np.sin(x[5])) + dhdx[
             5] * u4 + hx ** 3]  # LHS of the CBF constraint
    lbg = [0]  # Constraint bounds
    ubg = [inf]
    qp = {'x': u, 'f': V, 'g': vertcat(*g)}  # Formulate the QP
    solver = qpsol('solver', 'qpoases', qp, {'sparse': True, 'error_on_fail': False})  # Solve the QP
    sol = solver(lbx=lbu, ubx=ubu, lbg=lbg, ubg=ubg)  # Get the optimal solution
    u_mod = sol['x'].full().flatten()
    f_opt = sol['f'].full().flatten()
    return list(u_mod), list(f_opt), hx, dhdx


def CFTOC_NMPC(x_init, x_goal, T, N):
    # Solve the NMPC problem
    nx = 6
    nu = 4
    x1 = MX.sym('x1')  # Declare model variables
    x2 = MX.sym('x2')
    x3 = MX.sym('x3')
    x4 = MX.sym('x4')
    x5 = MX.sym('x5')
    x6 = MX.sym('x6')
    u1 = MX.sym('u1')
    u2 = MX.sym('u2')
    u3 = MX.sym('u3')
    u4 = MX.sym('u4')
    x = vertcat(x1, x2, x3, x4, x5, x6)
    u = vertcat(u1, u2, u3, u4)
    xdot = vertcat(u1 * cos(x3), u1 * sin(x3), u2, u3 * cos(x6), u3 * sin(x6), u4)  # Model equations
    if use_human_data:
        L = 5 * (x1 - x_goal[0]) ** 2 + 15 * (x2 - x_goal[1]) ** 2 + 1 * (x3 - x_goal[2]) ** 2 + \
            5 * (x4 - x_goal[3]) ** 2 + 15 * (x5 - x_goal[4]) ** 2 + 1 * (x6 - x_goal[5]) ** 2 + \
            1 * u1 ** 2 + 0.1 * u2 ** 2 + 1 * u3 ** 2 + 0.1 * u4 ** 2  # Objective term
    else:
        L = 5 * (x1 - x_goal[0]) ** 2 + 2 * (x2 - x_goal[1]) ** 2 + 1 * (x3 - x_goal[2]) ** 2 + \
            5 * (x4 - x_goal[3]) ** 2 + 2 * (x5 - x_goal[4]) ** 2 + 1 * (x6 - x_goal[5]) ** 2 + \
            1 * u1 ** 2 + 0.1 * u2 ** 2 + 1 * u3 ** 2 + 0.1 * u4 ** 2  # Objective term
    # Formulate discrete time dynamics using fixed step Runge-Kutta 4 integrator
    M = 4  # RK4 steps per interval
    DT = T / N / M
    f = Function('f', [x, u], [xdot, L])
    X0 = MX.sym('X0', nx)
    U = MX.sym('U', nu)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT / 2 * k1, U)
        k3, k3_q = f(X + DT / 2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    # "Lift" initial conditions
    Xk = MX.sym('X0', nx)
    w += [Xk]
    lbw += x_init
    ubw += x_init
    w0 += x_init
    # Formulate the NLP
    for k in range(N):
        Uk = MX.sym('U_' + str(k), nu)  # New NLP variable for the control
        w += [Uk]
        lbw += u_lb
        ubw += u_ub
        w0 += [0, 0, 0, 0]
        Fk = F(x0=Xk, p=Uk)  # Integrate till the end of the interval
        Xk_end = Fk['xf']
        J = J + Fk['qf']
        Xk = MX.sym('X_' + str(k + 1), nx)  # New NLP variable for state at end of interval
        w += [Xk]
        lbw += [-inf, -inf, -inf, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, inf, inf]
        w0 += [0, 0, 0, 0, 0, 0]
        g += [Xk_end - Xk]  # Add equality constraint
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}  # Create an NLP solver
    solver = nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)  # Solve the NLP
    w_opt = sol['x'].full().flatten()
    u_opt = list(w_opt[6:10])
    return u_opt


# The following functions are used in the constructive CBF [Squires et al]
def b(px, theta):
    return px - (v / w) * np.sin(theta)


def c(py, theta):
    return py + (v / w) * np.cos(theta)


def db(x):
    return b(x[0], x[2]) - b(x[3], x[5])


def dc(x):
    return c(x[1], x[2]) - c(x[4], x[5])


def compute_A2(x):
    A21, theta21 = phasor_add(
        -2 * db(x) * (v / w), 2 * db(x) * (v / w), x[2] + np.pi / 2, x[5] + np.pi / 2
    )
    A22, theta22 = phasor_add(-2 * dc(x) * (v / w), 2 * dc(x) * (v / w), x[2], x[5])
    A2, theta2 = phasor_add(A21, A22, theta21, theta22)
    return A2


def phasor_add(A1, A2, theta1, theta2):
    # Compute the phasor addition of two cosines
    A3 = np.sqrt((A1 * np.cos(theta1) + A2 * np.cos(theta2)) ** 2
                 + (A1 * np.sin(theta1) + A2 * np.sin(theta2)) ** 2)
    if A1 * np.cos(theta1) + A2 * np.cos(theta2) == 0:
        theta3 = np.sign(A1 * np.sin(theta1) + A2 * np.sin(theta2)) * pi / 2
    elif A1 * np.cos(theta1) + A2 * np.cos(theta2) > 0:
        theta3 = np.arctan((A1 * np.sin(theta1) + A2 * np.sin(theta2)) /
                           (A1 * np.cos(theta1) + A2 * np.cos(theta2)))
    else:
        theta3 = np.arctan((A1 * np.sin(theta1) + A2 * np.sin(theta2)) /
                           (A1 * np.cos(theta1) + A2 * np.cos(theta2))) + np.pi
    return A3, theta3


class PRNG(object):

    def __init__(self, seed):
        self.key = random.PRNGKey(seed)

    def next(self):
        k1, k2 = random.split(self.key)
        self.key = k1
        return k2


def alpha(x):
    # The class-K function
    return x ** 3


def pytorch_dynamics_f(x):
    return torch.zeros((6,))


def pytorch_dynamics_g(x: Tensor):
    zero_block = torch.zeros((x.shape[0], 3, 2), dtype=x.dtype, device=x.device)
    zero_one_block = torch.zeros((x.shape[0], 3, 1), dtype=x.dtype, device=x.device)
    zero_one_block[:, -1] = 1.
    zero = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    r1 = torch.vstack((torch.cos(x[:, 2]), torch.sin(x[:, 2]), zero)).T[:, :, None]
    r2 = torch.vstack((torch.cos(x[:, 5]), torch.sin(x[:, 5]), zero)).T[:, :, None]

    top = torch.concat((r1, zero_one_block, zero_block), axis=2)
    bottom = torch.concat((zero_block, r2, zero_one_block), axis=2)

    ret = torch.concat((top, bottom), axis=1)
    return ret


# Plot the closed-loop trajectories
def plot_closed_loop_multiple(X_cl_plot):
    fz = 20
    plt.clf()
    plt.figure(figsize=(6, 6))
    NUM_TRAJ = len(X_cl_plot)
    cm = plt.get_cmap('gist_rainbow')
    for i in range(NUM_TRAJ):
        Xcl = X_cl_plot[i,]
        for t in range(len(Xcl)):
            tgrid = [ts * k for k in range(t)]
            plt.plot(Xcl[:t, 0], Xcl[:t, 1], '-', color=cm(1. * i / NUM_TRAJ), linewidth=1.5)
            plt.plot(Xcl[:t, 3], Xcl[:t, 4], ':', color=cm(1. * i / NUM_TRAJ), linewidth=1.5)
            plt.plot(Xcl[0, 3], Xcl[0, 4], 'o', color=cm(1. * i / NUM_TRAJ), markersize=10)
            plt.plot(Xcl[0, 0], Xcl[0, 1], 'd', color=cm(1. * i / NUM_TRAJ), markersize=12)
    plt.xlabel('$p_x$', fontsize=1.2 * fz)
    plt.ylabel('$p_y$', fontsize=1.2 * fz)
    plt.rc('xtick', labelsize=fz)
    plt.rc('ytick', labelsize=fz)
    plt.legend(['agent a', 'agent b'], fontsize=0.8 * fz)
    # plt.xticks(np.arange(-2.5, 2.5+0.1, 0.5))
    # plt.yticks(np.arange(-2.0, 2.0+0.1, 0.5))
    plt.xlim([-2.0, 2.0])
    plt.ylim([-2.0, 2.0])
    # plt.axis('equal')
    plt.grid()
    plt.savefig('airplane_traj.png', dpi=300)
    # plt.show()


# A NN is represented as a list of [(W_1, b_1), ..., (W_n, b_n)]
# where n is the number of layers
def pytorch_h_model(inputs, params):
    # The L Lipchtiz implementation
    # return SequentialLipschitz(input_size=params[0][0].shape[0], output_size=1, hidden_sizes=[32, 64, 32],
    #                            scalable_lipschitz=True)

    return Sequential(Linear(6, 32),
                      ReLU(), Linear(32, 64), ReLU(), Linear(64, 32), ReLU(), Linear(32, 1))


def pytorch_r_with_input(h_model, x, u):
    y_pred = h_model(x)

    dh, = autograd.grad(y_pred, x,
                        grad_outputs=y_pred.data.new(y_pred.shape).fill_(1),
                        create_graph=True)

    df = torch.einsum("i,ni->n", torch.zeros((6,)), dh)
    df = df.reshape(-1, 1)

    dynam_g = pytorch_dynamics_g(x).transpose(2, 1)
    dynam_dg = torch.einsum("nji,ni->nj", dynam_g, dh)

    dg = torch.einsum("ni,ni->n", dynam_dg, u)
    dg = dg.reshape(-1, 1)

    a = alpha(h_model(x))

    return df + dg + a


def pytorch_loss(h_model, x_constraint, x_expert_unsafe, u_constraint, x_boundary, x_safe, x_unsafe,
                 safe_value, unsafe_value, lam_constraint, lam_boundary, lam_safe, lam_unsafe, gamma, writer, epoch):
    # loss_funct = lambda x: x
    loss_funct = torch.relu
    # loss_funct = torch.nn.ELU()

    if x_boundary is not None:
        boundary_cost = torch.square(h_model(x_boundary)).mean()
    else:
        boundary_cost = 0.0

        # try to enforce h(x) >= safe_value on safe set

    safe_cost = loss_funct(safe_value - h_model(x_safe)).mean() + loss_funct(
        safe_value / 10 - h_model(x_constraint)).mean()

    # try to enforce h(x) <= -unsafe_value on unsafe set
    if x_unsafe is not None:
        unsafe_cost = loss_funct(h_model(x_unsafe) + unsafe_value).mean() + loss_funct(
            h_model(x_expert_unsafe) + unsafe_value / 10).mean()
    else:
        unsafe_cost = 0.0

    # try to enforce the CBF inequality constraint
    constraint_cost = loss_funct(-pytorch_r_with_input(h_model, x_constraint, u_constraint) + gamma).mean() + \
                      0.001 * torch.square(pytorch_r_with_input(h_model, x_constraint, u_constraint)).mean()
    # constraint_cost *= 0

    # Seems to be a normalize cost for the weights, you can forgo this as the Lipschitz model helps
    # reduce the risk of overfitting

    # param_cost = torch.sum(torch.square(ravel_pytree(params)[0]))

    writer.add_scalar("loss/constraint", constraint_cost, epoch)
    writer.add_scalar("loss/boundary", boundary_cost, epoch)
    writer.add_scalar("loss/safe", safe_cost, epoch)
    writer.add_scalar("loss/unsafe", unsafe_cost, epoch)

    return (
            lam_constraint * constraint_cost + lam_boundary * boundary_cost + lam_safe * safe_cost
            + lam_unsafe * unsafe_cost
        # + lam_param * param_cost # Overfitting loss, you can add this back in if you want later on
    )


def pytorch_training(init_params, num_batches, x_constraint, x_expert_unsafe, u_constraint, u_expert_unsafe,
                     boundary_points,
                     safe_points, unsafe_points, safe_value, unsafe_value, lam_constraint, lam_boundary, lam_safe,
                     lam_unsafe, lam_param, gamma,
                     use_save=True):
    lam_constraint_batch = lam_constraint * num_batches
    lam_boundary_batch = lam_boundary * num_batches
    lam_safe_batch = lam_safe * num_batches
    lam_unsafe_batch = lam_unsafe * num_batches

    device = 'cpu'

    MODEL_PATH_FILE = '1nn.pth'

    net = pytorch_h_model(x_constraint, init_params)
    net = net.to(device)

    if os.path.exists(MODEL_PATH_FILE):
        if use_save:
            net.load_state_dict(torch.load(MODEL_PATH_FILE))
            return net

    # net = torch.compile(net, mode='max-autotune-no-cudagraphs')

    x_constraint = torch.tensor(np.array(x_constraint), requires_grad=True, device=device)
    x_expert_unsafe = torch.tensor(np.array(x_expert_unsafe), requires_grad=True, device=device)
    u_constraint = torch.tensor(np.array(u_constraint), requires_grad=True, device=device)
    u_expert_unsafe = torch.tensor(np.array(u_expert_unsafe), requires_grad=True, device=device)

    if boundary_points is None:
        boundary_points = None
    else:
        boundary_points = torch.tensor(np.array(boundary_points), requires_grad=True)
    safe_points = torch.tensor(np.array(safe_points), requires_grad=True)

    if unsafe_points is None:
        unsafe_points = None
    else:
        unsafe_points = torch.tensor(np.array(unsafe_points), requires_grad=True)

    optimizer = Adam(net.parameters(), lr=1e-3)

    summary_writer = SummaryWriter()

    summary_writer.add_graph(net, x_constraint)

    scheduler = CosineAnnealingWarmRestarts(optimizer, 500)
    scheduler2 = ReduceLROnPlateau(optimizer, patience=500)

    logging_interval = 100

    for i in range(int(1e5)):
        optimizer.zero_grad()

        loss_h = pytorch_loss(net, x_constraint, x_expert_unsafe, u_constraint, boundary_points,
                              safe_points, unsafe_points, safe_value, unsafe_value, lam_constraint , lam_boundary ,
                              lam_safe,
                              lam_unsafe, gamma, summary_writer, i)
        # loss_lipschitz = net.get_lipschitz()
        if callable(getattr(net, "get_lipschitz", None)):
            loss_lipschitz = net.get_lipschitz()**2
        else:
            loss_lipschitz = 0.0

        total_loss = loss_h + loss_lipschitz / 16 * 0.5

        total_loss.backward()
        optimizer.step()

        scheduler.step()
        scheduler2.step(total_loss)

        summary_writer.add_scalar('loss/total', total_loss, i)
        summary_writer.add_scalar('loss/h_function', loss_h, i)
        summary_writer.add_scalar('loss/lipschitz', loss_lipschitz, i)
        summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i)

        if i % logging_interval == 0:
            for name, param in net.named_parameters():
                if len(param.shape) == 1 and param.shape[0] == 1:
                    summary_writer.add_scalar(name, param, i)
                else:
                    summary_writer.add_histogram(name, param, i)

        print(
            "epoch", (i + 1), "| loss",
            total_loss.item(), "| loss_h",
            loss_h.item(), "| loss_l",
            loss_lipschitz.item() if loss_lipschitz != 0.0 else 0.0, "| elapsed",
            time.time() - start_time, "seconds"
        )

    with torch.no_grad():
        x = torch.vstack([x_constraint, x_expert_unsafe, unsafe_points])

        summary_writer.add_embedding(x, metadata=net(x))

    print("Saving Network")
    torch.save(net.state_dict(), "1nn.pth")
    print("Finished Saving Network")

    return net


if __name__ == '__main__':
    T = 1.0  # prediction horizon of MPC
    N = 10  # number of prediction steps of MPC
    ts = T / N  # sampling interval
    u_lb = [0.1, -1.0, 0.1, -1.0]  # lower bounds of the control inputs
    u_ub = [1.0, 1.0, 1.0, 1.0]  # upper bounds of the control inputs
    Ds = 0.5  # safe distance
    v = 0.1  # parameter used in true CBF
    w = 1.0  # parameter used in true CBF

    use_human_data = False
    sf = 1.

    regenerate_data = False

    save_directory = 'airplane_example/datasets/CBF-MPC'

    if not (os.path.exists(f"{save_directory}/X_train.npy") and os.path.exists(
            f"{save_directory}/U_train.npy")) or regenerate_data:
        # Option 2: Generate expert demonstrations using CBF-MPC
        # First generate a set of initial and the corresponding goal states
        X_start_t, X_goal_t = generate_init_condition(R=0.35, num_points=100)
        X_start, X_goal = X_start_t, X_goal_t
        X_start_t, X_goal_t = generate_init_condition(R=0.4, num_points=100)
        X_start, X_goal = np.vstack((X_start, X_start_t)), np.vstack((X_goal, X_goal_t))
        X_start_t, X_goal_t = generate_init_condition(R=0.45, num_points=100)
        X_start, X_goal = np.vstack((X_start, X_start_t)), np.vstack((X_goal, X_goal_t))
        X_start_t, X_goal_t = generate_init_condition(R=0.5, num_points=100)
        X_start, X_goal = np.vstack((X_start, X_start_t)), np.vstack((X_goal, X_goal_t))
        px_rel = X_start[:, 0] - X_start[:, 3]
        py_rel = X_start[:, 1] - X_start[:, 4]
        plt.figure(figsize=(10, 5))
        plt.plot(px_rel, py_rel, 'k.')
        plt.axis('equal')
        plt.xlabel('$p_{x,r}$')
        plt.ylabel('$p_{y,r}$')
        plt.grid()
        plt.savefig('samples.png', dpi=300)

        # Option 2 cont'd: Now generate expert demonstrations using CBF-MPC (takes about 15-20 minutes on Colab)
        X_train_raw, U_train_raw = generate_training_data(X_start, X_goal, N_sim=60)
        X_train = np.array(X_train_raw[0,])
        U_train = np.array(U_train_raw[0,])
        for i in range(1, len(X_train_raw)):
            X_train = np.vstack((X_train, X_train_raw[i,]))
            U_train = np.vstack((U_train, U_train_raw[i,]))
        h_train = np.array([ZCBF(x) for x in X_train])
        np.save(f'{save_directory}/X_train.npy', X_train)
        np.save(f'{save_directory}/U_train.npy', U_train)
        np.save(f'{save_directory}/h_train.npy', h_train)
    else:
        X_train = np.load(f"{save_directory}/X_train.npy")
        U_train = np.load(f'{save_directory}/U_train.npy')
        # h_train = np.load(f'{save_directory}/h_train.npy')

    # Define the model for the CBF candidate. We use a fully connected two layer NN.
    n_inputs = 6  # state dimension
    n_outputs = 1  # scalar valued
    n_hidden = 64

    # Initialize NN parameters
    rng = PRNG(5433)
    init_params = [[
        random.normal(rng.next(), shape=(n_inputs, n_hidden)),
        random.normal(rng.next(), shape=(n_hidden,))
    ], [
        random.normal(rng.next(), shape=(n_hidden, n_hidden * 2)),
        random.normal(rng.next(), shape=(n_hidden * 2,))
    ], [
        random.normal(rng.next(), shape=(n_hidden * 2, n_hidden)),
        random.normal(rng.next(), shape=(n_hidden,))
    ],
        [
            random.normal(rng.next(), shape=(n_hidden, n_outputs)),
            random.normal(rng.next(), shape=(n_outputs,))
        ]]

    # Augment the training set with safe and unsafe state samples
    # Reduce size of the training dataset (optional)
    if use_human_data:
        X_train, U_train = get_ring_data(X_train, U_train, Ds, 4 * Ds, 1)
    else:
        X_train = X_train[0::2]
        U_train = U_train[0::2]

    # Split expert states into safe ones outside the N ring and unsafe ones inside the N ring (optional)
    if use_human_data:
        width = Ds / 6  # width of the N ring
    else:
        width = 0.05  # width of the N ring
    print('Safe expert states outside N ring:')
    X_expert_safe, U_expert_safe = get_ring_data(X_train, U_train, Ds + width + 1e-5, 1e6, 1)
    print('Unsafe expert states inside N ring:')
    X_expert_unsafe, U_expert_unsafe = get_ring_data(X_train, U_train, Ds, Ds + width, 1)

    if use_human_data:
        # Safe samples (inside the geometric safe set)
        n_safe_samples = 10000
        safe_points = np.vstack((
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-1100, maxval=1100),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-1100, maxval=1100),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=0.0, maxval=3.1415),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-1100, maxval=1100),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-1100, maxval=1100),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=0.0, maxval=3.1415)
        )).T
        safe_points, _ = get_ring_data(safe_points, None, 450, 750, 0)
        print('Total number of safe samples:', safe_points.shape[0])

        # Unsafe samples (inside the N ring)
        n_angle_grid = 10
        n_unsafe_samples = 200
        angle_grid_a = list(np.linspace(0, -2 * pi, n_angle_grid).T)
        angle_grid_b = list(np.linspace(pi, -pi, n_angle_grid).T)
        unsafe_points_s_far = np.vstack((
            random.uniform(rng.next(), shape=(20000,), minval=-Ds,
                           maxval=Ds), random.uniform(rng.next(), shape=(20000,), minval=-Ds, maxval=Ds),
            random.uniform(rng.next(), shape=(20000,), minval=0.0, maxval=np.pi),
            random.uniform(rng.next(), shape=(20000,), minval=-Ds,
                           maxval=Ds), random.uniform(rng.next(), shape=(20000,), minval=-Ds, maxval=Ds),
            random.uniform(rng.next(), shape=(20000,), minval=0.0, maxval=np.pi)
        )).T
        unsafe_points_s_far, _ = get_ring_data(unsafe_points_s_far, None, 0, 0.8 * Ds, verbose=0)
        unsafe_points_s_far = unsafe_points_s_far[0:n_unsafe_samples // n_angle_grid, :]

        unsafe_points_s_near = np.vstack((
            random.uniform(rng.next(), shape=(20000,), minval=-Ds - 5 * width, maxval=Ds + 5 * width),
            random.uniform(rng.next(), shape=(20000,), minval=-Ds - 5 * width, maxval=Ds + 5 * width),
            random.uniform(rng.next(), shape=(20000,), minval=0.0, maxval=np.pi),
            random.uniform(rng.next(), shape=(20000,), minval=-Ds - 5 * width, maxval=Ds + 5 * width),
            random.uniform(rng.next(), shape=(20000,), minval=-Ds - 5 * width, maxval=Ds + 5 * width),
            random.uniform(rng.next(), shape=(20000,), minval=0.0, maxval=np.pi)
        )).T
        unsafe_points_s_near, _ = get_ring_data(unsafe_points_s_near, None, Ds, Ds + width, 0)
        unsafe_points_s_near = unsafe_points_s_near[0:40, :]
        unsafe_points_s = np.vstack((unsafe_points_s_far, unsafe_points_s_near))
        for i in range(n_angle_grid):
            unsafe_points_t = np.vstack((
                unsafe_points_s[:, 0], unsafe_points_s[:, 1],
                angle_grid_a[i] * jnp.ones_like(unsafe_points_s[:, 0]), unsafe_points_s[:, 3],
                unsafe_points_s[:, 4], angle_grid_b[i] * jnp.ones_like(unsafe_points_s[:, 0])
            )).T
            if i == 0:
                unsafe_points = unsafe_points_t
            else:
                unsafe_points = np.vstack((unsafe_points, unsafe_points_t))
        print(unsafe_points.shape[0], 'unsafe points obtained')
        if X_expert_unsafe.shape[0] > 0:
            unsafe_points = np.vstack((unsafe_points, X_expert_unsafe))
        print('Total number of unsafe points:', unsafe_points.shape[0])
    else:
        # Safe samples (inside the geometric safe set)
        n_safe_samples = 4000
        safe_points = np.vstack((
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-2.5, maxval=2.5),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-2.5, maxval=2.5),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=0.0, maxval=3.1415),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-2.5, maxval=2.5),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=-2.5, maxval=2.5),
            random.uniform(rng.next(), shape=(n_safe_samples // 2,), minval=0.0, maxval=3.1415)
        )).T
        safe_points_ham = Hammersly()
        random_seed = 0

        bounds = [(-2.5, 2.5), (-2.5, 2.5), (0., np.pi), (-2.5, 2.5), (-2.5, 2.5), (0, np.pi)]

        safe_points = np.array(safe_points_ham.generate(bounds, n_safe_samples // 2, random_state=random_seed))

        safe_points, _ = get_ring_data(
            safe_points, None, 1.5, 2.5, 0
        )  # also add all x_constraint to safe samples
        print('Total number of safe samples:', safe_points.shape[0])

        # Unsafe samples (inside the N ring)
        n_angle_grid = 10
        n_unsafe_samples = 5000
        angle_grid_a = list(np.linspace(0, -2 * pi, n_angle_grid).T)
        angle_grid_b = list(np.linspace(pi, -pi, n_angle_grid).T)
        unsafe_points_s_far = np.vstack((
            random.uniform(rng.next(), shape=(20000,), minval=-Ds,
                           maxval=Ds), random.uniform(rng.next(), shape=(20000,), minval=-Ds, maxval=Ds),
            random.uniform(rng.next(), shape=(20000,), minval=0.0, maxval=np.pi),
            random.uniform(rng.next(), shape=(20000,), minval=-Ds,
                           maxval=Ds), random.uniform(rng.next(), shape=(20000,), minval=-Ds, maxval=Ds),
            random.uniform(rng.next(), shape=(20000,), minval=0.0, maxval=np.pi)
        )).T

        bounds = [(-Ds, Ds), (-Ds, Ds), (0., np.pi), (-Ds, Ds), (-Ds, Ds), (0, np.pi)]

        unsafe_points_s_far = np.array(safe_points_ham.generate(bounds, 20000, random_state=random_seed))

        unsafe_points_s_far, _ = get_ring_data(unsafe_points_s_far, None, 0, 0.4, verbose=0)
        unsafe_points_s_far = unsafe_points_s_far[0:n_unsafe_samples // n_angle_grid, :]
        unsafe_points_s = unsafe_points_s_far
        for i in range(n_angle_grid):
            unsafe_points_t = np.vstack((
                unsafe_points_s[:, 0], unsafe_points_s[:, 1],
                angle_grid_a[i] * jnp.ones_like(unsafe_points_s[:, 0]), unsafe_points_s[:, 3],
                unsafe_points_s[:, 4], angle_grid_b[i] * jnp.ones_like(unsafe_points_s[:, 0])
            )).T
            if i == 0:
                unsafe_points = unsafe_points_t
            else:
                unsafe_points = np.vstack((unsafe_points, unsafe_points_t))
        print(unsafe_points.shape[0], 'unsafe points obtained')
        if X_expert_unsafe.shape[0] > 0:
            unsafe_points = np.vstack((unsafe_points, X_expert_unsafe))
        print('Total number of unsafe points:', unsafe_points.shape[0])

    # Convert the training data into JAX format
    x_constraint = jnp.vstack([X for X in X_expert_safe])
    u_constraint = jnp.vstack([U for U in U_expert_safe])
    x_safe = jnp.vstack([X for X in safe_points])
    x_unsafe = jnp.vstack([X for X in unsafe_points])
    x_expert_unsafe = jnp.vstack([X for X in X_expert_unsafe])
    u_expert_unsafe = jnp.vstack([U for U in U_expert_unsafe])

    # Plot the training data in relative positions
    fs = 18  # fontsize
    px_rel_constr = x_constraint[:, 0] - x_constraint[:, 3]
    py_rel_constr = x_constraint[:, 1] - x_constraint[:, 4]
    px_rel_expert_unsafe = x_expert_unsafe[:, 0] - x_expert_unsafe[:, 3]
    py_rel_expert_unsafe = x_expert_unsafe[:, 1] - x_expert_unsafe[:, 4]
    px_rel_safe = x_safe[:, 0] - x_safe[:, 3]
    py_rel_safe = x_safe[:, 1] - x_safe[:, 4]
    px_rel_unsafe = x_unsafe[:, 0] - x_unsafe[:, 3]
    py_rel_unsafe = x_unsafe[:, 1] - x_unsafe[:, 4]
    plt.figure(figsize=(7, 7))
    plt.plot(px_rel_constr / sf, py_rel_constr / sf, 'g.')
    plt.plot(px_rel_safe / sf, py_rel_safe / sf, 'b.')
    plt.plot(px_rel_expert_unsafe / sf, py_rel_expert_unsafe / sf, 'r.')
    plt.plot(px_rel_unsafe / sf, py_rel_unsafe / sf, 'r.')
    plt.xlabel('$p_{x,r}$', fontsize=fs)
    plt.ylabel('$p_{y,r}$', fontsize=fs)
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    # plt.xticks(np.arange(-2.5, 2.5+0.1, 0.5))
    # plt.yticks(np.arange(-2.5, 2.5+0.1, 0.5))
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    # plt.axis('equal')
    # plt.legend(['expert trajectories','safe samples','unsafe samples'], fontsize = fs)
    plt.grid()
    plt.savefig('airplane_data.png', dpi=300)
    plt.show()

    rng = PRNG(5433)
    start_time = time.time()
    params = init_params
    block_size = 1000
    num_blocks = 50
    num_epochs = block_size * num_blocks
    num_batches = 1
    safe_value = 1.5
    unsafe_value = 0.5
    lam_constraint = 15.0
    lam_boundary = 1.0
    # lam_safe = 15.0
    lam_safe = 2.0
    # lam_unsafe = 15.0
    lam_unsafe = 2.0
    lam_param = 0.1
    gamma = 0.05

    h_model = pytorch_training(init_params, num_batches, x_constraint, x_expert_unsafe, u_constraint, u_expert_unsafe,
                               None,
                               x_safe, x_unsafe, safe_value, unsafe_value, lam_constraint, lam_boundary, lam_safe,
                               lam_unsafe, lam_param,
                               gamma, use_save=False)
    h_model = h_model.eval()

    # Evaluate and plot the CBF at the training points
    X_draw = X_train[:, :]
    fs = 16  # fontsize
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    if use_human_data:
        rel_x_dist = X_draw[:, 0] / sf - X_draw[:, 3] / sf
        rel_y_dist = X_draw[:, 1] / sf - X_draw[:, 4] / sf
        hxl = np.array([h_model(x) for x in X_draw])
        ax.scatter3D(rel_x_dist, rel_y_dist, hxl, c=hxl, cmap='Blues', alpha=1.0)
        plt.legend(['learned CBF'], fontsize=1.2 * fs)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        # ax.set_zticks([-1.0,0,1.0,2.0])
        ax.set_xlim([-2.1, 2.1])
        ax.set_ylim([-2.1, 2.1])
        # ax.set_zlim([-0.2, 0.8])
    else:
        rel_x_dist = X_draw[:, 0] - X_draw[:, 3]
        rel_y_dist = X_draw[:, 1] - X_draw[:, 4]

        with torch.no_grad():
            hxl = h_model(torch.from_numpy(X_draw).type(torch.float))
            ax.scatter3D(rel_x_dist, rel_y_dist, hxl, c=hxl, cmap='Blues', alpha=1.0)
            hx = np.array([ZCBF(x) for x in X_draw])
            ax.scatter3D(rel_x_dist, rel_y_dist, hx, c=hx, cmap='Reds', alpha=0.1)
        plt.legend(['learned CBF', 'constructive CBF'], fontsize=1.2 * fs)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        # ax.set_zticks([-1.0,0,1.0,2.0])
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        # ax.set_zlim([-0.2, 0.8])
    plt.grid()
    ax.set_xlabel('$p_{x,r}$', fontsize=1.2 * fs)
    ax.set_ylabel('$p_{y,r}$', fontsize=1.2 * fs)
    # ax.set_zlabel('$h(x)$',    fontsize = fs)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)
    ax.zaxis.set_tick_params(labelsize=fs)
    plt.axis('equal')
    plt.savefig('airplane_CBFs.png', dpi=300)
    plt.show()

    # Safe closed-loop control with the learned CBF
    if use_human_data:
        T = 20.
        N = 10
        ts = T / N
        u_lb = [0.3, -0.05, 0.3, -0.05]
        u_ub = [3.0, 0.05, 3.0, 0.05]
        Ds = 100
        X_start, X_goal = generate_init_condition(R=200.0, num_points=24)
    else:
        X_start, X_goal = generate_init_condition(R=1.0, num_points=24)
    X_cl_LCBF = []  # store the closed-loop trajectories
    min_sep_set = []  # store the minimum separation between the two airplanes
    # idx_set = [0, 17, 6, 21, 2, 4, 19]  # the initial heading indices we used in our paper
    idx_set = [ 0]  # the initial heading indices we used in our paper
    for idx in idx_set:
        x_init = list(X_start[idx, :])
        x_goal = list(X_goal[idx, :])
        Xcl, Ucl, sep_min_cl = safe_closed_loop(
            x_init, x_goal, N_sim=100, learning=1, plot=1, early_stop=0
        )
        X_cl_LCBF.append(Xcl[:-1, :] / sf)
        min_sep_set.append(sep_min_cl / sf)

    print('safe distance required:', Ds / sf)
    print('minimal separations:', np.float16(min_sep_set))

    plot_closed_loop_multiple(np.array(X_cl_LCBF))
