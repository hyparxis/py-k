from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

import autograd.numpy as np
from autograd import jacobian
from autograd import grad

import time
import random
from functools import partial

from vis import draw_arm, draw_loop

EPS = 1e-3

def fk(q):
    x, y = 0.0, 0.0
    for i in range(len(q)):
        ang = np.array(0, dtype=float)
        for j in range(i + 1):
            ang += np.pi - q[j]
        x += l[i] * np.cos(ang)
        y += l[i] * np.sin(ang)

    return np.array([x, y])

def jacobian_ik(goal, q, inverse=True):
    lr = 0.1

    fk_jacobian = jacobian(fk)

    while np.any(np.abs(goal - fk(q)) > EPS):
        # get end effector Jacobian using automatic differentiation
        J = fk_jacobian(q)

        # pseudoinvert jacobian, can also just do transpose
        if inverse:
            Jinv = np.linalg.pinv(J)
        else:
            Jinv = J.T

        # calculate distance
        distance = goal - fk(q)

        q += lr * Jinv @ distance

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw_arm(q, l, c)

        pygame.display.flip()
        pygame.time.wait(1)


def gd_ik(goal, q, nesterov=False):
    n_itr = 0

    # 2 norm
    cost = lambda q: 1/2 * (goal - fk(q)).T @ (goal - fk(q))
    
    grad_cost = grad(cost)

    # momentum
    nu = 0.9
    
    # learning rate
    alpha = 0.01

    v = 0
    while np.any(np.abs(cost(q)) > EPS):
        if nesterov:
            v = nu * v + alpha * grad_cost(q - nu * v)
            q -= v
        else:
            q -= alpha * grad_cost(q)

        draw_arm(q, l, c)

        n_itr += 1
        #print("loss: ", cost(q))

    print(f"number of iterations: {n_itr}")

def program(goal, q):
    #jacobian_ik(goal, q, inverse=False)
    gd_ik(goal, q, nesterov=True)


def loop_cb(*args, **kwargs):
    draw_arm(q, l, c)


if __name__ == "__main__":
    global q, l, c

    n = int(input("Number of links: "))

    # linkage lengths
    l = [1] * n

    # linkage colors
    c = [tuple(random.uniform(0, 1) for c in range(3)) for _ in range(n)]

    # linkage generalized coordinates 
    q = np.random.uniform(0, 1, n)

    # callbacks take globals; maybe not neccessary 
    draw_loop(
        loop_cb=lambda *args, **kwargs: draw_arm(q, l, c),
        click_cb=lambda *args, **kwargs: gd_ik(args[0], q, True)
    )
