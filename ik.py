import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import torch

import numpy as np
from itertools import tee

import autograd.numpy as np
from autograd import jacobian

import time

# PARAMETERS
l = [1, 1, 1]
c = [(1,0,0),(0,1,0),(0,0,1)] # r, g, b
q = np.array([np.pi/2, np.pi/2, np.pi/2], dtype=float)


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
    px, py, pz = gluUnProject(x, y, 0)

    return (10 * px, 10 * py)

def main(fun):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    znear, zfar, dist = 1, 10, 10

    #mult = (2 * zfar * znear) / (znear - zfar) * dist

    gluPerspective(45, (display[0]/display[1]), znear, zfar)
    glTranslatef(0.0, 0.0, -dist)
    glLineWidth(5)


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                x, y = mouse_unproject(x, display[1] - y)
                
                x = float(input("x: "))
                y = float(input("y: "))

                print(x, y)
                print(fk(q))
                fun(np.array([x, y], dtype=float), q)
                print(fk(q))
    
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

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

    # put each line in global cartesian coordinates
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

def program(goal, q):
    lr = 0.1

    fk_jacobian = jacobian(fk)

    while np.any(goal - fk(q) > 0.00001):
        # get end effector Jacobian using automatic differentiation
        J = fk_jacobian(q)

        # pseudoinvert jacobian
        Jinv = np.linalg.pinv(J)

        # calculate "cost"
        cost = goal - fk(q)

        q += lr * Jinv @ cost

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw_arm(q, l, c)

        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    main(program)