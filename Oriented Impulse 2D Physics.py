import pygame
from pygame import gfxdraw
import math
import random
import numpy as np
import numpy.typing as npt

pygame.init()

a = pygame.image.load("C:\\Users\\marcu\\OneDrive\\Pictures\\Saved Pictures\\unnamed.jpg")
pygame.display.set_icon(a)
pygame.display.set_caption("2D Physics")

clock = pygame.time.Clock()

bg = (0,0,0)

pygame.init()

width = 1280
height = 658

FPS = 120

screen = pygame.display.set_mode([width,height])

a = True
b = False
randColour = (255,255,255)

camera = [0,0,1]

mx,my = (0,0)

mxo,myo = (0,0)

antiGravity = False
if not antiGravity:
    gravity = 0.09
else:
    gravity = 0
slop = 0.01
percent = 0.5
recalculations = 0

mode = 0

def dotproduct(p1,p2):
    return p1[0]*p2[0]+p1[1]*p2[1]

def crossproduct(input1,input2):
    if (isinstance(input1,list))and(isinstance(input2,list)):
        return input1[0]*input2[1] - input1[1]*input2[0]

    elif (isinstance(input1,list))and((isinstance(input2,int))or(isinstance(input2,float))):
        return [input2*input1[1],-input2*input1[0]]

    elif ((isinstance(input1,int))or(isinstance(input1,float))and(isinstance(input2,list))):
        return [-input1*input2[1],input1*input2[0]]


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def matXvec(mat,vec):
    return [dotproduct(mat[0],vec),dotproduct(mat[1],vec)]

def rotate(points,orientation):
    mat = [[math.cos(-orientation),-math.sin(-orientation)],
           [math.sin(-orientation),math.cos(-orientation)]]

    newPoints = []
    for point in points:
        newPoints.append(matXvec(mat,point))

    return newPoints

def mat(orientation):
    return [[math.cos(-orientation),-math.sin(-orientation)],[math.sin(-orientation),math.cos(-orientation)]]

def explode_xy(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0])
        yl.append(xy[i][1])
    return xl,yl

def shoelace_area(x_list,y_list):
    a1,a2=0,0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l=abs(a1-a2)/2
    return l

def polyArea(points):
    xy = [tuple(x) for x in points]
    xy_e = explode_xy(xy)

    A=shoelace_area(xy_e[0],xy_e[1])
    return A

def polyRadiusSquared(points):
    polyRadii = []
    for point in points:
        polyRadii.append(point[0]**2+point[1]**2)
    return max(polyRadii)

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

def dist(x1,y1,x2,y2):
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def linePoint(p1,p2,x,y):
    d1 = dist(x,y,p1[0],p1[1])
    d2 = dist(x,y,p2[0],p2[1])

    lineLen = dist(p1[0],p1[1],p2[0],p2[1])

    buffer = 0.0000001

    if (d1+d2 >= lineLen-buffer)and(d1+d2 <= lineLen+buffer):
        return True
    return False

def get_closest_point(x1,y1,x2,y2,a,b):
    if x1==x2: return [x1,b]
    if y1==y2: return [a,y1]
    m1 = (y2-y1)/(x2-x1)
    m2 = -1/m1
    x = (m1*x1-m2*a+b-y1) / (m1-m2)
    y = m2*(x-a)+b
    return [x,y]

def findClosestPoint(A,p):
    nextIndex = 0
    bestClosest = None
    bestDistance = float("+inf")
    for i in range(len(A)):
        nextIndex = i+1
        if nextIndex == len(A):
            nextIndex = 0

        p1 = A[i]
        p2 = A[nextIndex]

        closest = get_closest_point(p1[0],p1[1],p2[0],p2[1],p[0],p[1])

        d = (closest[0]-p[0])**2+(closest[1]-p[1])**2

        if d<bestDistance:
            bestDistance = d
            bestClosest = closest

    return bestClosest,bestDistance**0.5

def polygonPoint(A,p1,radius):
    nextIndex = 0
    collision = False
    for i in range(len(A)):
        nextIndex = i+1
        if nextIndex == len(A):
            nextIndex = 0

        vc = A[i]
        vn = A[nextIndex]

        if (((vc[1] > p1[1] and vn[1] < p1[1]) or (vc[1] < p1[1] and vn[1] > p1[1]))and(p1[0] < (vn[0]-vc[0])*(p1[1]-vc[1])/(vn[1]-vc[1])+vc[0])):
            collision = not collision

    if collision:
        closest,penetration = findClosestPoint(A,p1)
        penetration += radius
        return True,closest,penetration,closest,closest
    else:
        return False,None,None,None,None
    
def edgeCirclecollide(circle,p1,p2):
    distX = p1[0]-p2[0]
    distY = p1[1]-p2[1]
    length = (distX**2+distY**2)

    dot = (((circle[0]-p1[0])*(p2[0]-p1[0]))+((circle[1]-p1[1])*(p2[1]-p1[1])))/length

    closestX = p1[0] + (dot * (p2[0]-p1[0]))
    closestY = p1[1] + (dot * (p2[1]-p1[1]))

    onSegment = linePoint(p1,p2,closestX,closestY)
    if not onSegment:
        if dotproduct([closestX-p1[0],closestY-p1[1]],[p2[0]-p1[0],p2[1]-p1[1]])>0:
            closestX = p2[0]
            closestY = p2[1]
        else:
            closestX = p1[0]
            closestY = p1[1]

    distX = closestX - circle[0]
    distY = closestY - circle[1]
    distance = distX**2+distY**2

    if distance<=(circle[2]**2):
        distance = distance**0.5
        normal = [distX/distance,distY/distance]
        penetration = circle[2]-distance
        radiusa = [closestX,closestY]
        radiusb = [closestX,closestY]
        return True,normal,penetration,radiusa,radiusb

    return False,None,None,None,None

def PolyCirclecollide(A,B):
    inside,normal,penetration,radiusa,radiusb = polygonPoint(A,[B[0],B[1]],B[2])
    if inside:
        penetration = penetration + B[2]
        d = (normal[0]**2+normal[1]**2)**0.5
        normal = [-normal[0]/d,-normal[1]/d]
        return True,normal,penetration,radiusa,radiusb
    
    nextIndex = 0
    for i in range(len(A)):
        nextIndex = i+1
        if nextIndex == len(A):
            nextIndex = 0

        vc = A[i]
        vn = A[nextIndex]

        collided,normal,penetration,radiusa,radiusb = edgeCirclecollide(B,vc,vn)

        if collided:
            return True,normal,penetration,radiusa,radiusb

    return False,None,None,None,None

def centers_displacement(p1, p2):
    c1 = np.mean(np.array(p1), axis=0)
    c2 = np.mean(np.array(p2), axis=0)
    return c2 - c1

def generateConvex(c,n: int) -> npt.NDArray[np.float64]:
    X_rand, Y_rand = np.sort(np.random.random(n)*c), np.sort(np.random.random(n)*c)
    X_new, Y_new = np.zeros(n), np.zeros(n)

    last_true = last_false = 0
    for i in range(1, n):
        if i != n - 1:
            if random.getrandbits(1):
                X_new[i] = X_rand[i] - X_rand[last_true]
                Y_new[i] = Y_rand[i] - Y_rand[last_true]
                last_true = i
            else:
                X_new[i] = X_rand[last_false] - X_rand[i]
                Y_new[i] = Y_rand[last_false] - Y_rand[i]
                last_false = i
        else:
            X_new[0] = X_rand[i] - X_rand[last_true]
            Y_new[0] = Y_rand[i] - Y_rand[last_true]
            X_new[i] = X_rand[last_false] - X_rand[i]
            Y_new[i] = Y_rand[last_false] - Y_rand[i]

    np.random.shuffle(Y_new)
    vertices = np.stack((X_new, Y_new), axis=-1)
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

    vertices = np.cumsum(vertices, axis=0)

    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] += ((x_max - np.min(vertices[:, 0])) / 2) - x_max
    vertices[:, 1] += ((y_max - np.min(vertices[:, 1])) / 2) - y_max

    return vertices

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       return False,None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    x1 = [i[0] for i in line1]
    x2 = [i[0] for i in line2]
    y1 = [i[1] for i in line1]
    y2 = [i[1] for i in line2]

    tolerance = 0.01

    _1mix = min(x1)-tolerance
    _1max = max(x1)+tolerance
    _2mix = min(x2)-tolerance
    _2max = max(x2)+tolerance
    _1miy = min(y1)-tolerance
    _1may = max(y1)+tolerance
    _2miy = min(y2)-tolerance
    _2may = max(y2)+tolerance

    if (x>_1mix)and(x>_2mix)and(x<_1max)and(x<_2max)and(y>_1miy)and(y>_2miy)and(y<_1may)and(y<_2may):    
        return True,[x, y]
    else:
        return False,None

def getContact(A,B):
    pointsA = rotate(rotate(A.pointsFromOg,A.orientation),-B.orientation)
    for point in pointsA: point = [point[0]+A.xPos-B.xPos,point[1]+A.yPos-B.yPos]
    pointsB = B.pointsFromOg

    intersections = []
    
    for i in range(len(pointsA)):
        nextIndex = i+1
        if nextIndex == len(pointsA):
            nextIndex = 0

        edgeA = [pointsA[i],pointsA[nextIndex]]

        for j in range(len(pointsB)):
            nextIndex = j+1
            if nextIndex == len(pointsB):
                nextIndex = 0

            edgeB = [pointsB[j],pointsB[nextIndex]]

            intersection,point = line_intersection(edgeA,edgeB)
            if intersection:
                if not point in intersections:
                    intersections.append(point)


    if len(intersections)==0:
        return [0,0]
    else:
        return rotate([intersections[0]],B.orientation)[0]
        return rotate([[sum([i[0] for i in intersections])/len(intersections),sum([i[1] for i in intersections])/len(intersections)]],B.orientation)[0]
    

def atan2(x,y):
    if(x==0)and(y==0):
        return 0
    elif(x==0):
        if y>0:
            return math.pi*0.5
        else:
            return math.pi*1.5
    else:
        return math.atan2(y,x)
    

def CirclePolyCollide(centre,radius,poly1):
    pass

#circle: type 0
class circle:
    def __init__(self,centrex,centrey,radius,orientation,xVel,yVel,angVel,torque,density,restitution,friction,colour):
        self.centrex = centrex
        self.centrey = centrey
        self.radius = radius
        self.orientation = orientation
        self.xVel = xVel
        self.yVel = yVel
        self.angVel = angVel
        self.torque = torque
        self.density = density
        self.restitution = restitution
        self.friction = friction
        self.colour = colour

    def type(self):
        return 0

    def invMass(self):
        if self.density==0:
            return 0
        volume = (math.pi)*(self.radius**2)
        return 1/((self.density/100)*volume)

    def invMomentOfInertia(self):
        try:
            mass = 1/self.invMass()
        except ZeroDivisionError:
            mass = 0
        if mass == 0:
            return 0
        radius = self.radius
        return 1/(mass*(radius))

    def update(self):
        if self.invMass()!=0:
            self.yVel -= gravity
        self.angVel += self.torque * self.invMomentOfInertia()
        self.orientation += self.angVel
        self.centrex += self.xVel
        self.centrey += self.yVel

    def detect(self):
        collisions = []
        for thing in things:
            if (thing!=self)and(not(thing.invMass()==0 and self.invMass() == 0)):
                if thing.type()==0:
                    d = (thing.centrex-self.centrex)**2+(thing.centrey-self.centrey)**2
                    if d<((self.radius+thing.radius)**2):
                        d = d**0.5
                        if d!=0:
                            normal = [(thing.centrex-self.centrex)/d,(thing.centrey-self.centrey)/d]
                            penetration = self.radius + thing.radius - d
                        else:
                            normal = [1,0]
                            penetration = self.radius*2

                        radiusa = [-normal[0]*(self.radius-penetration),-normal[1]*(self.radius-penetration)]
                        radiusb = [normal[0]*(thing.radius-penetration),normal[1]*(thing.radius-penetration)]

                        try:
                            x = int((radiusa[0]+self.centrex+camera[0])/camera[2]+(width/2))
                            y = int(-(radiusa[1]+self.centrey-camera[1])/camera[2]+(height/2))
                            pygame.draw.circle(screen,(255,0,0),(x,y),int(10/camera[2]))
                        except:
                            pass
                            
                        collisions.append([thing,normal,penetration,radiusa,radiusb])

                if thing.type()==1:
                    pointsToCollide = rotate(thing.pointsFromOg,thing.orientation)
                    A = pointsToCollide
                    B = [self.centrex-thing.xPos,self.centrey-thing.yPos,self.radius]
                    
                    collided, normal, penetration, radiusa, radiusb = PolyCirclecollide(A,B)
                    if collided:
                        try:
                            x = int((radiusa[0]+thing.xPos+camera[0])/camera[2]+(width/2))
                            y = int(-(radiusa[1]+thing.yPos-camera[1])/camera[2]+(height/2))
                            pygame.draw.circle(screen,(255,0,0),(x,y),int(10/camera[2]))
                        except:
                            pass
                        collisions.append([thing,normal,penetration,radiusa,radiusb])
        
        return collisions

    def resolve(self,collisions):
        for collision in collisions:
            rv = [collision[0].xVel-self.xVel,collision[0].yVel-self.yVel]
            velAlongNormal = dotproduct(rv,collision[1])

            if velAlongNormal <= 0:
                e = min(collision[0].restitution,self.restitution)

                j = -(1+e) * velAlongNormal
                j /= (self.invMass()+collision[0].invMass()+((crossproduct(collision[3],collision[1])**2)*self.invMomentOfInertia())+((crossproduct(collision[4],collision[1]))**2)*thing.invMomentOfInertia())

                impulse = [collision[1][0]*j,collision[1][1]*j]

                self.xVel -= self.invMass()*impulse[0]
                self.yVel -= self.invMass()*impulse[1]
                collision[0].xVel += collision[0].invMass()*impulse[0]
                collision[0].yVel += collision[0].invMass()*impulse[1]

                rv = [collision[0].xVel-self.xVel,collision[0].yVel-self.yVel]
                dot = dotproduct(rv,collision[1])
                sub = [collision[1][0]*dot,collision[1][1]*dot]
                tangent = [rv[0]-sub[0],rv[1]-sub[1]]
                magnit = (tangent[0]**2+tangent[1]**2)**0.5
                if magnit != 0:
                    tangent = [tangent[0]/magnit,tangent[1]/magnit]

                    jt = -dotproduct(rv,tangent)
                    jt /= (self.invMass()+collision[0].invMass()+((crossproduct(collision[3],tangent)**2)*self.invMomentOfInertia())+((crossproduct(collision[4],tangent))**2)*thing.invMomentOfInertia())

                    mu = (self.friction**2+collision[0].friction**2)**0.5

                    frictionImpulse = []
                    if abs(jt)<j*mu:
                        frictionImpulse = [tangent[0]*jt,tangent[1]*jt]
                    else:
                        frictionImpulse = [tangent[0]*-j*mu,tangent[1]*-j*mu]

                    self.xVel -= self.invMass()*frictionImpulse[0]
                    self.yVel -= self.invMass()*frictionImpulse[1]
                    collision[0].xVel += collision[0].invMass()*frictionImpulse[0]
                    collision[0].yVel += collision[0].invMass()*frictionImpulse[1]
                    if self.density!=0:
                        self.angVel += self.invMomentOfInertia()*crossproduct(collision[3],frictionImpulse)
                    if collision[0].density!=0:
                        collision[0].angVel -= thing.invMomentOfInertia()*crossproduct(collision[4],frictionImpulse)
##                    if self.density!=0:
##                        self.angVel = ((abs(self.xVel)+abs(self.yVel))*math.sin(atan2(collision[3][1],collision[3][0])))/((collision[3][0]**2+collision[3][1]**2)**0.5)
##                    if collision[0].density!=0:
##                        collision[0].angVel = ((abs(thing.xVel)+abs(thing.yVel))*math.sin(atan2(collision[4][1],collision[4][0])))/((collision[4][0]**2+collision[4][1]**2)**0.5)
##                    self.angVel *= (1-mu)
##                    collision[0].angVel *= (1-mu)

            correction = (max(collision[2]-slop,0)/(self.invMass()+collision[0].invMass()))*percent
            correction = [collision[1][0]*correction,collision[1][1]*correction]

            self.centrex -= self.invMass() * correction[0]
            self.centrey -= self.invMass() * correction[1]

            if collision[0].type() == 0:
                collision[0].centrex += collision[0].invMass() * correction[0]
                collision[0].centrey += collision[0].invMass() * correction[1]
            elif collision[0].type() == 1:
                collision[0].xPos += collision[0].invMass()*correction[0]
                collision[0].yPos += collision[0].invMass()*correction[1]
            

    def draw(self):
        pygame.gfxdraw.aacircle(screen,int((self.centrex+camera[0])/camera[2]+(width/2)),int((-self.centrey+camera[1])/camera[2]+(height/2)),int(self.radius/camera[2]),self.colour)
        pygame.draw.aaline(screen,self.colour,(int((self.centrex+camera[0])/camera[2]+(width/2)),int((-self.centrey+camera[1])/camera[2]+(height/2))),(math.cos(self.orientation)*(self.radius/camera[2])+int((self.centrex+camera[0])/camera[2]+(width/2)),-math.sin(self.orientation)*(self.radius/camera[2])+int((-self.centrey+camera[1])/camera[2]+(height/2))))

#poly: type 1
class poly:
    def __init__(self,xPos,yPos,pointsFromOg,orientation,xVel,yVel,angVel,torque,density,restitution,friction,colour):
        self.xPos = xPos
        self.yPos = yPos
        self.pointsFromOg = pointsFromOg
        self.orientation = orientation
        self.xVel = xVel
        self.yVel = yVel
        self.angVel = angVel
        self.torque = torque
        self.density = density
        self.restitution = restitution
        self.friction = friction
        self.colour = colour

    def type(self):
        return 1

    def invMass(self):
        if self.density==0:
            return 0
        volume = polyArea(self.pointsFromOg)
        return 1/((self.density/100)*volume)

    def invMomentOfInertia(self):
        try:
            mass = (1/self.invMass())
        except ZeroDivisionError:
            mass = 0
        if mass==0:
            return 0
        radius = (polyRadiusSquared(self.pointsFromOg))**0.5
        return 1/(mass*(radius))

    def update(self):
        if self.invMass()!=0:
            self.yVel -= gravity
        self.angVel += self.torque * self.invMomentOfInertia()
        self.orientation += self.angVel
        self.xPos += self.xVel
        self.yPos += self.yVel

    def detect(self):
        collisions = []
        
        for thing in things:
            if (thing!=self)and(not(thing.invMass()==0 and self.invMass() == 0)):
                if thing.type()==0:
                    pointsToCollide = rotate(self.pointsFromOg,self.orientation)
                    A = pointsToCollide
                    B = [thing.centrex-self.xPos,thing.centrey-self.yPos,thing.radius]
                    
                    collided, normal, penetration, radiusa, radiusb = PolyCirclecollide(A,B)
                    if collided:
                        radiusb = [radiusb[0]+self.xPos-thing.centrex,radiusb[1]+self.yPos-thing.centrey]
                        normal = [-normal[0],-normal[1]]
                        try:
                            x = int((radiusa[0]+self.xPos+camera[0])/camera[2]+(width/2))
                            y = int(-(radiusa[1]+self.yPos-camera[1])/camera[2]+(height/2))
                            pygame.draw.circle(screen,(0,255,0),(x,y),int(10/camera[2]))
                        except:
                            pass
                        collisions.append([thing,normal,penetration,radiusa,radiusb])
                    
                if thing.type()==1:
                    poly1 = rotate(self.pointsFromOg,self.orientation)
                    for point in poly1:
                        point[0] = point[0]+self.xPos
                        point[1] = -(point[1]+self.yPos)
                    poly2 = rotate(thing.pointsFromOg,thing.orientation)
                    for point in poly2:
                        point[0] = point[0]+thing.xPos
                        point[1] = -(point[1]+thing.yPos)
                    collided,normal = PolyPolycollide(poly1,poly2)
                    if collided:
                        penetration = (normal[0]**2+normal[1]**2)**0.5
                        normal = [normal[0]/-penetration,normal[1]/penetration]
                        x = int((normal[0]*10+self.xPos+camera[0])/camera[2]+(width/2))
                        y = int(-(normal[1]*10+self.yPos-camera[1])/camera[2]+(height/2))
                        pygame.draw.line(screen,(0,0,255),((self.xPos+camera[0])/camera[2]+(width/2),-(self.yPos-camera[1])/camera[2]+(height/2)),(x,y),2)
                        radiusa = getContact(thing,self)
                        radiusb = getContact(self,thing)
                        try:
                            x = int((radiusa[0]+self.xPos+camera[0])/camera[2]+(width/2))
                            y = int(-(radiusa[1]+self.yPos-camera[1])/camera[2]+(height/2))
                            pygame.draw.circle(screen,(255,0,0),(x,y),int(10/camera[2]))
                        except:
                            pass
                        collisions.append([thing,normal,penetration,radiusa,radiusb])
        
        return collisions

    def resolve(self,collisions):
        for collision in collisions:
            rv = [collision[0].xVel-self.xVel,collision[0].yVel-self.yVel]
            velAlongNormal = dotproduct(rv,collision[1])

            if velAlongNormal <= 0:
                e = min(collision[0].restitution,self.restitution)

                j = -(1+e) * (velAlongNormal)
                j /= (self.invMass()+collision[0].invMass()+((crossproduct(collision[3],collision[1])**2)*self.invMomentOfInertia())+((crossproduct(collision[4],collision[1]))**2)*thing.invMomentOfInertia())

                impulse = [collision[1][0]*j,collision[1][1]*j]

                self.xVel -= self.invMass()*impulse[0]
                self.yVel -= self.invMass()*impulse[1]
                collision[0].xVel += collision[0].invMass()*impulse[0]
                collision[0].yVel += collision[0].invMass()*impulse[1]

                rv = [collision[0].xVel-self.xVel,collision[0].yVel-self.yVel]
                dot = dotproduct(rv,collision[1])
                sub = [collision[1][0]*dot,collision[1][1]*dot]
                tangent = [rv[0]-sub[0],rv[1]-sub[1]]
                magnit = (tangent[0]**2+tangent[1]**2)**0.5
                if magnit != 0:
                    tangent = [tangent[0]/magnit,tangent[1]/magnit]

                    jt = -dotproduct(rv,tangent)
                    jt /= (self.invMass()+collision[0].invMass()+((crossproduct(collision[3],tangent)**2)*self.invMomentOfInertia())+((crossproduct(collision[4],tangent))**2)*thing.invMomentOfInertia())

                    mu = (self.friction**2+collision[0].friction**2)**0.5

                    frictionImpulse = []
                    if abs(jt)<j*mu:
                        frictionImpulse = [tangent[0]*jt,tangent[1]*jt]
                    else:
                        frictionImpulse = [tangent[0]*-j*mu,tangent[1]*-j*mu]

                    self.xVel -= self.invMass()*frictionImpulse[0]
                    self.yVel -= self.invMass()*frictionImpulse[1]
                    collision[0].xVel += collision[0].invMass()*frictionImpulse[0]
                    collision[0].yVel += collision[0].invMass()*frictionImpulse[1]
                    if self.density!=0:
                        self.angVel += self.invMomentOfInertia()*crossproduct(collision[3],frictionImpulse)
                    if collision[0].density!=0:
                        collision[0].angVel -= thing.invMomentOfInertia()*crossproduct(collision[4],frictionImpulse)
##                    if thing.type()==0:
##                        if self.density!=0:
##                            self.angVel = ((abs(self.xVel)+abs(self.yVel))*math.sin(atan2(collision[3][1],collision[3][0])))/((collision[3][0]**2+collision[3][1]**2)**0.5)
##                        if collision[0].density!=0:
##                            collision[0].angVel = ((abs(thing.xVel)+abs(thing.yVel))*math.sin(atan2(collision[4][1],collision[4][0])))/((collision[4][0]**2+collision[4][1]**2)**0.5)
##                    self.angVel *= (1-mu)
##                    collision[0].angVel *= (1-mu)

            correction = (max(collision[2]-slop,0)/(self.invMass()+collision[0].invMass()))*percent
            correction = [collision[1][0]*correction,collision[1][1]*correction]

            self.xPos -= self.invMass()*correction[0]
            self.yPos -= self.invMass()*correction[1]

            if collision[0].type() == 0:
                collision[0].centrex += collision[0].invMass() * correction[0]
                collision[0].centrey += collision[0].invMass() * correction[1]
            elif collision[0].type() == 1:
                collision[0].xPos += collision[0].invMass()*correction[0]
                collision[0].yPos += collision[0].invMass()*correction[1]

    def draw(self):
        pointsToDraw = rotate(self.pointsFromOg,self.orientation)
        for point in pointsToDraw:
            point[0] = int((point[0]+self.xPos+camera[0])/camera[2]+(width/2))
            point[1] = int(-(point[1]+self.yPos-camera[1])/camera[2]+(height/2))
        pointsToDraw = [tuple(x) for x in pointsToDraw]
        pygame.gfxdraw.aapolygon(screen,pointsToDraw,self.colour)

things = [circle(0,300,200,0,0,0,0,0,0,0.4,0.2,(255,255,255)),poly(0,-9999999,[[9999999,9999999],[-9999999,9999999],[-9999999,-9999999],[9999999,-9999999]],0,0,0,0,0,0,0.7,0.2,(255,255,255)),poly(1500,1200,[[3000,50],[-1000,50],[-1000,-50],[3000,-50]],0.5,0,0,0,0,0,0.7,0.2,(255,255,255))]

ogLen = len(things)

shifted = False

running = True
while running:
    mx,my = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False                

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:
                if shifted:
                    mode -= 1
                    if mode == -1:
                        mode = 2
                else:
                    mode += 1
                    if mode == 3:
                        mode = 0
            if event.button == 4:
                oldZoom = 1/camera[2]
                camera[2]*=0.84674488777
                newZoom = 1/camera[2]
                
            elif event.button == 5:
                camera[2]*=1.18099325362

    screen.fill(bg)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        if mode == 0:
            things.append(circle((mx-(width/2))*camera[2]-camera[0]+random.randint(-1000,1000),(my-(height/2))*-camera[2]+camera[1]+random.randint(-1000,1000),random.randint(100,1000),0,0,0,0,0,(0 if shifted else 0.2),0.7,0.2,(random.randint(150,255),random.randint(150,255),random.randint(150,255))))
        elif mode == 1:
            randLength = random.randint(50,200)
            randHeight = random.randint(50,200)
            randOffx = random.randint(-1000,1000)
            randOffy = random.randint(-1000,1000)
            left,top,right,bottom = (-randLength/2,+randHeight/2,randLength/2,-randHeight/2)
            xPos = (mx-(width/2))*camera[2]-camera[0]+randOffx
            yPos = (my-(height/2))*-camera[2]+camera[1]+randOffy
            points = [[left,top],[left,bottom],[right,bottom],[right,top]]
            things.append(poly(xPos,yPos,points,0,0,0,0,0,(0 if shifted else 0.2),0.7,0.2,(random.randint(150,255),random.randint(150,255),random.randint(150,255))))
        elif mode == 2:
            randPoly = generateConvex(random.randint(100,1000),random.randint(5,20))
            randOffx = random.randint(-1000,1000)
            randOffy = random.randint(-1000,1000)
            xPos = (mx-(width/2))*camera[2]-camera[0]+randOffx
            yPos = (my-(height/2))*-camera[2]+camera[1]+randOffy
            things.append(poly(xPos,yPos,randPoly,0,0,0,0,0,(0 if shifted else 0.2),0.7,0.2,(random.randint(150,255),random.randint(150,255),random.randint(150,255))))


    if keys[pygame.K_r]:
        if shifted:
            things = things[:ogLen]
        else:
            things = [i for i in things if i.density==0]

    if keys[pygame.K_LSHIFT]:
        shifted = True
    else:
        shifted = False

    left,middle,right = pygame.mouse.get_pressed(num_buttons=3)
    if left:
        if a:
            pygame.mouse.get_rel()
            a = False
        else:
            x,y = pygame.mouse.get_rel()
            camera[0] += x*camera[2]
            camera[1] += y*camera[2]
    else:
        a = True

    if right:
        if not b:
            b = True
            mxo,myo = pygame.mouse.get_pos()
            randColour = (random.randint(150,255),random.randint(150,255),random.randint(150,255))
        if b:
            pygame.draw.aaline(screen,randColour,(mxo,myo),(mx,my))

    else:
        if b:
            randObject = mode
            if randObject == 0:
                things.append(circle((mxo-(width/2))*camera[2]-camera[0],(myo-(height/2))*-camera[2]+camera[1],random.randint(100,1000),(atan2(-mxo+mx,myo-my) if shifted else 0),(0 if shifted else (mxo*camera[2]-mx*camera[2])/100),(0 if shifted else (-myo*camera[2]+my*camera[2])/100),0,0,(0 if shifted else 0.2),0.7,0.2,randColour))
            elif randObject == 1:
                randLength = random.randint(50,200)
                randHeight = random.randint(50,200)
                left,top,right,bottom = (-randLength/2,+randHeight/2,randLength/2,-randHeight/2)
                xPos = (mxo-(width/2))*camera[2]-camera[0]
                yPos = (myo-(height/2))*-camera[2]+camera[1]
                points = [[left,top],[left,bottom],[right,bottom],[right,top]]
                things.append(poly(xPos,yPos,points,(atan2(-mxo+mx,myo-my) if shifted else 0),(0 if shifted else (mxo*camera[2]-mx*camera[2])/100),(0 if shifted else (-myo*camera[2]+my*camera[2])/100),0,0,(0 if shifted else 0.2),0.7,0.2,randColour))
            elif randObject == 2:
                randPoly = generateConvex(random.randint(100,1000),random.randint(5,20))
                xPos = (mxo-(width/2))*camera[2]-camera[0]
                yPos = (myo-(height/2))*-camera[2]+camera[1]
                things.append(poly(xPos,yPos,randPoly,(atan2(-mxo+mx,myo-my) if shifted else 0),(0 if shifted else (mxo*camera[2]-mx*camera[2])/100),(0 if shifted else (-myo*camera[2]+my*camera[2])/100),0,0,(0 if shifted else 0.2),0.7,0.2,randColour))

        b = False

    for thing in things:
        thing.update()
        for i in range(1+recalculations):
            collisions = thing.detect()
            thing.resolve(collisions)
        try:
            thing.draw()
        except:
            del thing
    
    pygame.display.update()

    clock.tick(FPS)

pygame.quit()
