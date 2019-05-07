import random 
import autograd.numpy as np
from autograd import grad
from autograd import jacobian

from vis import draw_init, draw_arm, draw_loop

EPS = 1e-3

def fk(q, l):
    x, y = 0.0, 0.0
    for i in range(len(q)):
        ang = np.array(0, dtype=float)
        for j in range(i + 1):
            ang += np.pi - q[j]
        x += l[i] * np.cos(ang)
        y += l[i] * np.sin(ang)

    return np.array([x, y])

def nsp_ik(goal, q, nesterov=True):
    def cost(q):
        objective = (goal - fk(q[0:3], l1)) 
        constraint = np.pi - q[2]

        return objective.T @ objective + \
               constraint.T * constraint

    def g(q):
        return fk(q[0:2], l1) - fk(q[3:], l2)

    grad_cost = grad(cost)

    constraint_jac = jacobian(g)

    alpha = 0.01
    nu = 0.9
    max_itr = 100

    v, n_itr = 0, 0
    while np.any(np.abs(cost(q)) > EPS):

        # get constraint jacobian
        G = constraint_jac(q)

        Ginv = np.linalg.pinv(G)

        I = np.identity(Ginv.shape[0])

        # matrix to projects into the null space of the constraint jacobian
        nsp = (I - G.T @ Ginv.T).T

        if nesterov:
            v = nu * v + alpha * grad_cost(q - nu * v)
            q -= nsp @ v
        else:
            q -= nsp @ (alpha * grad_cost(q))

        draw_arm(q[0:3], l1, c1, swap=False)
        draw_arm(q[3:], l2, c2, clear=False)

        n_itr += 1
        if n_itr > max_itr:
            break

    

def gd_ik(goal, q, nesterov=True):
    max_itr = 100

    # end effector of q1 must be at goal,
    # second link of q2 must be at second link of q1
    # q2 must be 0
    def cost(q):
        objective = (goal - fk(q[0:3], l1))
        constraint1 = fk(q[0:2], l1) - fk(q[3:], l2)
        constraint2 = 2 * np.pi - q[2]

        return objective.T @ objective + \
               constraint1.T @ constraint1 + \
               constraint2.T * constraint2

    grad_cost = grad(cost)

    # momentum
    nu = 0.9
    
    # learning rate
    alpha = 0.1

    v, n_itr = 0, 0
    while np.any(np.abs(cost(q)) > EPS):
        if nesterov:
            v = nu * v + alpha * grad_cost(q - nu * v)
            q -= v
        else:
            q -= alpha * grad_cost(q)

        draw_arm(q[0:3], l1, c1, swap=False)
        draw_arm(q[3:], l2, c2, clear=False)

        n_itr += 1
        if n_itr > max_itr:
            break
        #print("loss: ", cost(q))

    print(f"number of iterations: {n_itr}")

def program(goal):
    #gd_ik(goal, q, False)
    nsp_ik(goal, q)

def render():
    draw_arm(q[0:3], l1, c1, swap=False)
    draw_arm(q[3:], l2, c2, clear=False)

if __name__ == "__main__":
    global q1, q2, l1, l2, c1, c2

    l1 = [1, .2, .8]
    l2 = [.2, 1]

    q = np.random.uniform(0, 1, len(l1 + l2))

    q = np.array(
        [8.57922587e+00,
         1.43566805e+00,
         5.98935607e-03, 
         6.90821247e+00,
         4.81874893e+00]
    )

    c1 = [tuple(random.uniform(0, 1) for c in range(3)) for _ in range(len(l1))]
    c2 = [tuple(random.uniform(0, 1) for c in range(3)) for _ in range(len(l2))]

    draw_init()

    draw_arm(q[0:3], l1, c1)
    draw_arm(q[3:], l2, c2)

    draw_loop(
        loop_cb=lambda *args, **kwargs: render(),
        click_cb=lambda *args, **kwargs: program(args[0])
    )