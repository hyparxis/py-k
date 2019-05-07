import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from util import pairwise

_init = False
display = (800, 600)

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

def draw_arm(q, l, c, clear=True, swap=True):
    if clear:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

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
    
    if swap:
        pygame.display.flip()
        pygame.time.wait(1)

def draw_init():
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    znear, zfar, dist = 1, 10, 10

    gluPerspective(45, (display[0]/display[1]), znear, zfar)
    glTranslatef(0.0, 0.0, -dist)
    glLineWidth(5)

    global _init
    _init = True

def draw_loop(loop_cb=None, click_cb=None):
    while True:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                x, y = mouse_unproject(x, display[1] - y)
                #draw_point(x, y)

                if click_cb is not None:
                    click_cb(np.array([x, y], dtype=float), **globals())

        if loop_cb is not None:
            loop_cb(**globals())

        pygame.display.flip()
        pygame.time.wait(10)