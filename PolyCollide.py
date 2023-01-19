import numpy as np
import pygame
from pygame import gfxdraw

pygame.init()

screen = pygame.display.set_mode([1280,658])

def edges_of(vertices):
    edges = []
    N = len(vertices)

    for i in range(N):
        edge = vertices[(i + 1)%N] - vertices[i]
        edges.append(edge)

    return edges

def orthogonal(v):
    return np.array([-v[1], v[0]])

def is_separating_axis(o, p1, p2):
    min1, max1 = float('+inf'), float('-inf')
    min2, max2 = float('+inf'), float('-inf')

    for v in p1:
        projection = np.dot(v, o)

        min1 = min(min1, projection)
        max1 = max(max1, projection)

    for v in p2:
        projection = np.dot(v, o)

        min2 = min(min2, projection)
        max2 = max(max2, projection)

    if max1 >= min2 and max2 >= min1:
        d = min(max2 - min1, max1 - min2)
        d_over_o_squared = d/np.dot(o, o) + 1e-10
        pv = d_over_o_squared*o
        return False, pv
    else:
        return True, None

def PolyPolycollide(p1, p2):
    p1 = [np.array(v, 'float64') for v in p1]
    p2 = [np.array(v, 'float64') for v in p2]

    edges = edges_of(p1)
    edges += edges_of(p2)
    orthogonals = [orthogonal(e) for e in edges]

    push_vectors = []
    for o in orthogonals:
        separates, pv = is_separating_axis(o, p1, p2)

        if separates:
            return False, None
        else:
            push_vectors.append(pv)

    mpv =  min(push_vectors, key=(lambda v: np.dot(v, v)))

    d = centers_displacement(p1, p2)
    if np.dot(d, mpv) > 0:
        mpv = -mpv

    return True, mpv


def centers_displacement(p1, p2):
    c1 = np.mean(np.array(p1), axis=0)
    c2 = np.mean(np.array(p2), axis=0)
    return c2 - c1

poly1 = [[0,0],[-100,0],[-100,-200],[-25,-100]]
poly2 = [[0,0],[-100,0],[-100,-100]]

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        for point in poly1:
            point[0]-=1

    elif keys[pygame.K_RIGHT]:
        for point in poly1:
            point[0]+=1

    if keys[pygame.K_UP]:
        for point in poly1:
            point[1]-=1

    elif keys[pygame.K_DOWN]:
        for point in poly1:
            point[1]+=1

    screen.fill((0,0,0))

    poly1draw = []
    poly2draw = []
    for point in poly1:
        poly1draw.append([point[0]+640,point[1]+329])

    for point in poly2:
        poly2draw.append([point[0]+640,point[1]+329])

    pygame.gfxdraw.aapolygon(screen,[tuple(x) for x in poly1draw],(255,255,255))
    pygame.gfxdraw.aapolygon(screen,[tuple(x) for x in poly2draw],(255,255,255))

    pygame.display.update()

    clock.tick(60)

    collided,mpv = PolyPolycollide(poly1,poly2)

    if collided:
        for point in poly1:
            point[0]+=mpv[0]
            point[1]+=mpv[1]

pygame.quit()

