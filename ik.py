import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import torch

import numpy as np
from itertools import tee

import autograd.numpy as np
from autograd import jacobian
from autograd import grad

import time

import random

# PARAMETERS
n = int(input("Number of links: "))
l = [1] * n
c = [tuple(random.uniform(0, 1) for c in range(3)) for _ in range(n)]
q = np.random.uniform(0, 1, n)

#q = np.array([0] * n, dtype=float)

EPS = 1e-3

def draw_point(x, y, rad=1):
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glPointSize(rad)

    glBegin(GL_POINTS)

    glColor3f(1, 0, 0)
    glVertex3f(x, y, 0)

    glEnd()
    

def draw_line(p1, p2, color=None):
    glBegin(GL_LINES)

    if color is not None:
        glColor3f(*color)
    
    glVertex2f(p1[0], p1[1])

    glVertex2f(p2[0], p2[1])

    glEnd()   

def pol2car(r, theta):
    return r * np.array([np.cos(theta), np.sin(theta)])

def mouse_unproject(x, y):
    z = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    px, py, pz = gluUnProject(x, y, z)

    return (px, py)

def main(fun):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    znear, zfar, dist = 1, 10, 10

    gluPerspective(45, (display[0]/display[1]), znear, zfar)
    glTranslatef(0.0, 0.0, -dist)
    glLineWidth(5)


    while True:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                x, y = mouse_unproject(x, display[1] - y)
                draw_point(x, y)
                #print(x, y)
                # x = float(input("x: "))
                # y = float(input("y: "))

                # print(fk(q))
                fun(np.array([x, y], dtype=float), q)
                # print(fk(q))
    
        draw_arm(q, l, c)

        pygame.display.flip()
        pygame.time.wait(10)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def draw_arm(q, l, c):
    # zip into matrix of polar vectors
    model = np.array([list(a) for a in zip(l, q)])

    # convert generalized angles to global angles
    model[:, 1] = np.pi - model[:, 1]
    model[:, 1] = np.cumsum(model[:, 1])

    # convert to cartesian vectors, add root
    model = [[0, 0]] + [pol2car(*row) for row in model]

    # put each point in global cartesian coordinates
    model = np.cumsum(model, axis=0)

    for (i, line) in enumerate(pairwise(model)):
        draw_line(*line, color=c[i])

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

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw_arm(q, l, c)

        pygame.display.flip()
        pygame.time.wait(1)

        n_itr += 1
        #print("loss: ", cost(q))

    print(f"number of iterations: {n_itr}")

def program(goal, q):
    #jacobian_ik(goal, q, inverse=False)
    gd_ik(goal, q, nesterov=True)


if __name__ == "__main__":
    main(program)