#python -m cProfile -s tottime "C:\Users\marcu\OneDrive\Desktop\Python Scripts\Impulse 2D Physics\THE FINAL ORIENTED IMPULSE 2D PHYSICS.pyw"

import sys
import pygame
from pygame import gfxdraw
import math
import random

import polygenerator

def generateConvex(bounds,n):
    return [(p[0]*bounds-bounds/2,p[1]*bounds-bounds/2) for p in polygenerator.random_convex_polygon(n)]

def dist(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def length(v):
    return math.sqrt(v[0]**2+v[1]**2)

def com(p):
    return [sum(x)/len(x) for x in zip(*p)]

def rotate(p,a):
    cos = math.cos(a)
    sin = math.sin(a)
    return [p[0]*cos-p[1]*sin,p[0]*sin+p[1]*cos]

class Camera:
    def __init__(self,x=0,y=0,m=1):
        self.x = x
        self.y = y
        self.m = m
        self.xv = 0
        self.yv = 0
        self.mv = 0
        self.maxZoom = 25
        self.minZoom = 0.01
        self.cameraPanSpeed = 0.05
        self.cameraZoomSpeed = 0.02
        self.cameraPanFriction = 0.95
        self.cameraZoomFriction = 0.95

    def reset(self,x=0,y=0,m=1):
        self.x = x
        self.y = y
        self.m = m
        self.xv = 0
        self.yv = 0
        self.mv = 0

    def update(self,keys,scrolling,shift,dt):
        panSpeed = self.cameraPanSpeed*(shift+1)
        if keys[pygame.K_w]:
            self.yv += panSpeed
        elif keys[pygame.K_s]:
            self.yv -= panSpeed
        if keys[pygame.K_a]:
            self.xv -= panSpeed
        elif keys[pygame.K_d]:
            self.xv += panSpeed

        self.mv += scrolling*self.cameraZoomSpeed*self.m
            
        self.xv *= self.cameraPanFriction
        self.yv *= self.cameraPanFriction
        self.mv *= self.cameraZoomFriction
        self.x += dt*self.xv/self.m
        self.y += dt*self.yv/self.m
        self.m += self.mv
        if self.m > self.maxZoom:
            self.m = self.maxZoom
            self.mv = 0
        elif self.m < self.minZoom:
            self.m = self.minZoom
            self.mv = 0
        
    def toCameraSpace(self,coord):
        return [(coord[0]-self.x)*self.m + hwidth,(self.y-coord[1])*self.m + hheight]

    def toWorldSpace(self,coord):
        return [(coord[0] - hwidth)/self.m + self.x,-(coord[1] - hheight)/self.m + self.y]

def draw_polygon(surface,colour,points):
    try:
        p = [camera.toCameraSpace(point) for point in points]
        pygame.gfxdraw.aapolygon(surface,p,colour)
    except:
        pass

def draw_circle(surface,colour,x,y,r):
    try:
        pos = camera.toCameraSpace((x,y))
        l = r*camera.m
        pygame.gfxdraw.aacircle(surface,int(pos[0]),int(pos[1]),int(l),colour)
    except:
        pass

def draw_circle_with_line(surface,colour,r,x,y,a):
    try:
        pos = camera.toCameraSpace((x,y))
        l = r*camera.m
        pygame.gfxdraw.aacircle(surface,int(pos[0]),int(pos[1]),int(l),colour)
        pygame.draw.aaline(surface,colour,pos,(pos[0]+math.cos(-a)*l,pos[1]+math.sin(-a)*l))
    except:
        pass

def draw_circle_with_cross(surface,colour,r,x,y,a):
    try:
        pos = camera.toCameraSpace((x,y))
        l = r*camera.m
        pygame.gfxdraw.aacircle(surface,int(pos[0]),int(pos[1]),int(l),colour)

        a += math.pi/4

        cos = math.cos(-a)
        sin = math.sin(-a)

        l *= 0.8
        
        pygame.draw.aaline(surface,colour,(pos[0]-cos*l,pos[1]-sin*l),(pos[0]+cos*l,pos[1]+sin*l))
        pygame.draw.aaline(surface,colour,(pos[0]-sin*l,pos[1]+cos*l),(pos[0]+sin*l,pos[1]-cos*l))
    except:
        pass

def draw_point(surface,colour,point):
    try:
        pygame.draw.circle(surface,colour,camera.toCameraSpace(point))
    except:
        pass

def draw_line(surface,colour,p1,p2):
    try:
        pygame.draw.aaline(surface,colour,camera.toCameraSpace(p1),camera.toCameraSpace(p2))
    except:
        pass

def draw_rect(surface,colour,p1,p2):
    try:
        a = camera.toCameraSpace(p1)
        b = camera.toCameraSpace(p2)

        if a[0] > b[0]:
            a[0],b[0] = (b[0],a[0])

        if a[1] > b[1]:
            a[1],b[1] = (b[1],a[1])
        
        pygame.draw.rect(surface,colour,(a[0],a[1],b[0]-a[0],b[1]-a[1]),1)
    except:
        pass

def dataToPoints(P,x,y,a):
    sin = math.sin(a)
    cos = math.cos(a)

    return [[P[i][0]*cos-P[i][1]*sin+x,P[i][0]*sin+P[i][1]*cos+y] for i in range(len(P))]
    
def dot(v1,v2):
    return v1[0]*v2[0]+v1[1]*v2[1]

def getFurthest(p,n):
    bestProjection = float("-inf")
    bestVertex = None
    for i in range(len(p)):
        projection = dot(p[i],n)
        if projection > bestProjection:
            bestProjection = projection
            bestVertex = i

    return bestVertex

def getEdge(p,idx,cw):
    idx -= cw
    nex = 0 if idx == len(p)-1 else idx+1
    edge = (p[idx][0]-p[nex][0],p[idx][1]-p[nex][1])
    d = length(edge)
    return p[idx],p[nex],(edge[0]/d,edge[1]/d)

def is_separating_axis(o,p1,p2):
    min1 = float('inf')
    max1 = float('-inf')
    min2 = float('inf')
    max2 = float('-inf')

    for v in p1:
        projection = dot(v, o)

        if projection < min1: min1 = projection
        if projection > max1: max1 = projection

    for v in p2:
        projection = dot(v, o)

        if projection < min2: min2 = projection
        if projection > max2: max2 = projection

    if max1 >= min2 and max2 >= min1:
        a = max2 - min1
        b = max1 - min2
        if a<b: d = a
        else: d = b

        o_squared = o[0]**2+o[1]**2
        if o_squared == 0: d_over_o_squared = 1
        else: d_over_o_squared = d/o_squared
        
        pv = (o[0]*d_over_o_squared,o[1]*d_over_o_squared)
        return False, pv
    else:
        return True, None

def PolyPolycollide(p1,p2,c1,c2):
    mpv = None
    d = float("inf")
    for i in range(len(p1)):
        o = (p1[i][1]-p1[i-1][1],p1[i-1][0]-p1[i][0])
        separates, pv = is_separating_axis(o,p1,p2)

        if separates:
            return False, None
        else:
            pvdot = dot(pv,pv)
            if pvdot < d:
                mpv = pv
                d = pvdot

    for i in range(len(p2)):
        o = (p2[i][1]-p2[i-1][1],p2[i-1][0]-p2[i][0])
        separates, pv = is_separating_axis(o,p1,p2)

        if separates:
            return False, None
        else:
            pvdot = dot(pv,pv)
            if pvdot < d:
                mpv = pv
                d = pvdot

    d = (c1[0]-c2[0],c1[1]-c2[1])
    if dot(d,mpv) < 0:
        mpv = (-mpv[0],-mpv[1])

    return True, mpv

def clip(v1,v2,n,o):
    cp = []
    d1 = dot(n,v1) - o
    d2 = dot(n,v2) - o
    if d1 >= 0: cp.append(v1)
    if d2 >= 0: cp.append(v2)

    if d1*d2 < 0:
        e = (v2[0]-v1[0],v2[1]-v1[1])
        u = d1 / (d1-d2)
        e = (e[0]*u+v1[0],e[1]*u+v1[1])
        cp.append(e)

    return cp

def polyArea(P):
    return abs(sum([P[i-1][0]*P[i][1]-P[i-1][1]*P[i][0] for i in range(len(P))]))/2

def triangleInertia(p1,p2):
    return (p1[0]*p2[1]-p2[0]*p1[1])*(p1[0]**2+p1[1]**2+p2[0]**2+p2[1]**2+p1[0]*p2[0]+p1[1]*p2[1])

def momentOfInertia(P):
    return sum([triangleInertia(P[i-1],P[i]) for i in range(len(P))])/12

def linePoint(p1,p2,x,y):
    return 0 < (x-p1[0])*(p2[0]-p1[0])+(y-p1[1])*(p2[1]-p1[1]) < (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2

def get_closest_point(p1,p2,circle):
    distX = p1[0]-p2[0]
    distY = p1[1]-p2[1]
    length = (distX**2+distY**2)
    
    dp = (((circle[0]-p1[0])*(p2[0]-p1[0]))+((circle[1]-p1[1])*(p2[1]-p1[1])))/length

    closestX = p1[0] + (dp * (p2[0]-p1[0]))
    closestY = p1[1] + (dp * (p2[1]-p1[1]))

    onSegment = linePoint(p1,p2,closestX,closestY)

    if not onSegment:
        if dot([closestX-p1[0],closestY-p1[1]],[p2[0]-p1[0],p2[1]-p1[1]])>0:
            closestX = p2[0]
            closestY = p2[1]
        else:
            closestX = p1[0]
            closestY = p1[1]

    return (closestX,closestY)

def findClosestPoint(A,p):
    bestClosest = None
    bestDistance = float("+inf")
    for i in range(len(A)):
        p1 = A[i-1]
        p2 = A[i]

        closest = get_closest_point(p1,p2,p)

        d = (closest[0]-p[0])**2+(closest[1]-p[1])**2

        if d<bestDistance:
            bestDistance = d
            bestClosest = closest

    return bestClosest,math.sqrt(bestDistance)

def polygonPoint(A,p1,radius):
    collision = False
    for i in range(len(A)):
        vc = A[i-1]
        vn = A[i]

        if (((vc[1] > p1[1] and vn[1] < p1[1]) or (vc[1] < p1[1] and vn[1] > p1[1]))and(p1[0] < (vn[0]-vc[0])*(p1[1]-vc[1])/(vn[1]-vc[1])+vc[0])):
            collision = not collision

    if collision:
        closest,length = findClosestPoint(A,p1)
        normal = [(closest[0]-p1[0])/-length,(closest[1]-p1[1])/-length]
        return True,normal,length+radius,[closest]
    else:
        return False,None,None,None

    
def edgeCirclecollide(circle,p1,p2):
    distX = p1[0]-p2[0]
    distY = p1[1]-p2[1]
    length = (distX**2+distY**2)

    dp = (((circle[0]-p1[0])*(p2[0]-p1[0]))+((circle[1]-p1[1])*(p2[1]-p1[1])))/length

    closestX = p1[0] + (dp * (p2[0]-p1[0]))
    closestY = p1[1] + (dp * (p2[1]-p1[1]))

    onSegment = linePoint(p1,p2,closestX,closestY)
    if not onSegment:
        if dot([closestX-p1[0],closestY-p1[1]],[p2[0]-p1[0],p2[1]-p1[1]])>0:
            closestX = p2[0]
            closestY = p2[1]
        else:
            closestX = p1[0]
            closestY = p1[1]

    distX = closestX - circle[0]
    distY = closestY - circle[1]
    distance = distX**2+distY**2

    if distance<=(circle[2]**2):
        if distance == 0:
            return True,[0,-1],circle[2],[closestX,closestY]
        else:
            distance = math.sqrt(distance)
            normal = [distX/distance,distY/distance]
            penetration = circle[2]-distance
            return True,normal,penetration,[closestX,closestY]

    return False,None,None,None

def PolyCircleCollide(A,B):
    maxDist = float("-inf")
    bestNormal = None
    bestContact = None
    contacts = []
    
    for i in range(len(A)):
        vc = A[i-1]
        vn = A[i]

        if dot((vc[0]-B[0],vc[1]-B[1]),(vc[1]-vn[1],vn[0]-vc[0])) > 0:
            collided,normal,penetration,contact = edgeCirclecollide(B,vc,vn)

            if collided:
                if penetration > maxDist:
                    bestNormal = normal
                    bestContact = i
                    maxDist = penetration

                contacts.append(contact)

    if contacts: return True,bestNormal,maxDist,contacts

    return polygonPoint(A,[B[0],B[1]],B[2])


def AABBAABBCollide(A,B):
    return A[0] <= B[2] and B[0] <= A[2] and A[3] <= B[1] and B[3] <= A[1]

class Scene:
    def __init__(self,gravity=(0,-0.0098),wind=(0,0),linearAirFriction=0.01,QuadraticAirFriction=0.001,airFrictionMultiplier=1):
        self.gravity = gravity
        self.wind = wind
        self.linearAirFriction = linearAirFriction	
        self.QuadraticAirFriction = QuadraticAirFriction
        self.airFrictionMultiplier = airFrictionMultiplier

        self.objs = []
        
        self.springs = []
        self.fixjoints = []
        self.hinges = []
        self.axles = []
        self.thrusters = []
        self.tracers = []
        self.lasers = []
        
        self.collisions = []
        self.noCollisionGroups = set()

        self.paused = True

    def update(self):
        if not self.paused:
            for i in range(len(self.objs)):
                self.objs[i].collide(i)
                
            for collision in self.collisions:
                collision.resolve()

            self.collisions = []

            for spring in self.springs:
                spring.update()

            for hinge in self.hinges:
                hinge.update()

            for axle in self.axles:
                axle.update()

            for thruster in self.thrusters:
                thruster.update()

            for obj in self.objs:
                obj.move()

            for fixjoint in self.fixjoints:
                fixjoint.update()

    def draw(self):
        for obj in self.objs: obj.draw()
        for spring in self.springs: spring.draw()
        for hinge in self.hinges: hinge.draw()
        for fixjoint in self.fixjoints: fixjoint.draw()
        for axle in self.axles: axle.draw()
        for tracer in self.tracers: tracer.draw()
        for thruster in self.thrusters: thruster.draw()

class Collision:
    def __init__(self,A,B,contacts,normal,penetration):
        self.A = A
        self.B = B
        self.contacts = contacts
        self.normal = normal
        self.penetration = penetration

        scene.collisions.append(self)

    def resolve(self,slop=0.0198,percentage=0.25):
        A = self.A
        B = self.B
        
        if A.invMass+B.invMass == 0:
            return

        for contact in self.contacts:
            #draw_point(screen,(255,0,0),contact)
            
            normal = (-self.normal[0],-self.normal[1])
            ra = (contact[0]-A.x,contact[1]-A.y)
            rb = (contact[0]-B.x,contact[1]-B.y)
            rv = (B.xv - B.av*rb[1] - A.xv + A.av*ra[1], B.yv + B.av*rb[0] - A.yv - A.av*ra[0])

            contactVel = dot(rv,normal)

            if contactVel <= 0:
                raCrossN = ra[0]*normal[1]-ra[1]*normal[0]
                rbCrossN = rb[0]*normal[1]-rb[1]*normal[0]
                invMassSum = A.invMass+B.invMass + (raCrossN**2)*A.invInertia + (rbCrossN**2)*B.invInertia

                j = (-(((1 + min(A.restitution,B.restitution))*contactVel) / invMassSum) / len(self.contacts))

                impulse = (normal[0]*j,normal[1]*j)
                A.applyImpulse((-impulse[0],-impulse[1]),ra)
                B.applyImpulse(impulse,rb)

                rv = (B.xv - B.av*rb[1] - A.xv + A.av*ra[1], B.yv + B.av*rb[0] - A.yv - A.av*ra[0])

                t = (rv[0]-normal[0]*contactVel,rv[1]-normal[1]*contactVel)
                magnitude = length(t)
                if magnitude != 0:
                    t = (t[0]/magnitude,t[1]/magnitude)
                
                    jt = -((dot(rv,t) / invMassSum) / len(self.contacts))

                    sf = (A.sf+B.sf)/2
                    df = (A.df+B.df)/2

                    if jt != 0:
                        if abs(jt) < j*sf: tangentImpulse = (t[0]*jt,t[1]*jt)
                        else: tangentImpulse = (t[0] * -j * df, t[1] * -j * df)

                        A.applyImpulse((-tangentImpulse[0],-tangentImpulse[1]),ra)
                        B.applyImpulse(tangentImpulse,rb)
        
        if self.penetration > slop:
            mult = self.penetration*percentage
            correction = (self.normal[0]*mult, self.normal[1]*mult)

            s = A.invMass + B.invMass
            a = A.invMass/s
            b = B.invMass/s
            A.positionalCorrection((correction[0]*a,correction[1]*a))
            B.positionalCorrection((-correction[0]*b,-correction[1]*b))

class NoCollisionGroup:
    def __init__(self,*objs):
        self.objs = []
        for obj in objs:
            obj.noCollisionGroup = self
            self.objs.append(obj)

        scene.noCollisionGroups.add(self)

    def add(self,*objs):
        self.objs.extend(objs)
        for obj in objs: obj.noCollisionGroup = self

    def merge(self,other):
        for obj in other.objs: obj.noCollisionGroup = self
        scene.noCollisionGroups.remove(other)

    def close(self):
        for obj in self.objs: obj.noCollisionGroup = float("nan")
        scene.noCollisionGroups.remove(self)

class Circle:
    def __init__(self,r,x,y,a,xv=0,yv=0,av=0,density=1,restitution=0.8,sf=0.5,df=0.5,immovable=False,colour=(255,255,255),layer=1):
        self.r = r
        self.x = x
        self.y = y
        self.a = math.radians(a)
        self.immovable = immovable
        self.xv = 0 if self.immovable else xv
        self.yv = 0 if self.immovable else yv
        self.av = 0 if self.immovable else math.radians(av)
        self.invMass = 0 if self.immovable else density/(math.pi*(r**2))
        self.invInertia = 0 if self.immovable else 4/(math.pi*(r**4)*density)
        self.restitution = restitution
        self.sf = sf
        self.df = df
        self.colour = colour

        self.layer = layer
        self.noCollisionGroup = float("nan")

        self.updateData()

        scene.objs.append(self)

    def draw(self):
        draw_circle_with_line(screen,self.colour,self.r,self.x,self.y,self.a)
        #draw_rect(screen,(255,0,0),(self.AABB[0],self.AABB[1]),(self.AABB[2],self.AABB[3]))

    def updateData(self):
        self.AABB = (self.x-self.r,self.y+self.r,self.x+self.r,self.y-self.r)

    def collide(self,idx):   
        for i in range(idx+1,len(scene.objs)):
            obj = scene.objs[i]

            if self.invMass + obj.invMass == 0 or self.noCollisionGroup == obj.noCollisionGroup or self.layer & obj.layer == 0: continue

            if type(obj) == Circle:
                if not AABBAABBCollide(self.AABB,obj.AABB): continue

                d = (self.x-obj.x)**2+(self.y-obj.y)**2
                
                if d < (self.r+obj.r)**2:
                    if d == 0:
                        contact = (self.x,self.y)
                        pvn = (-1,0)
                        penetration = self.r

                        Collision(self,obj,[contact],pvn,penetration)
                        
                    else:
                        d = math.sqrt(d)
                        pvn = ((obj.x-self.x)/-d,(obj.y-self.y)/-d)

                        contact = (self.x-pvn[0]*self.r,self.y-pvn[1]*self.r)
                        
                        penetration = self.r+obj.r-d
                    
                        Collision(self,obj,[contact],pvn,penetration)
                        
            elif type(obj) == Polygon:
                B = obj.dataAsPoints

                if not AABBAABBCollide(self.AABB,obj.AABB): continue
                
                collided,pvn,penetration,contacts = PolyCircleCollide(B,[self.x,self.y,self.r])
                
                if collided:
                    pvn = (-pvn[0],-pvn[1])

                    Collision(self,obj,contacts,pvn,penetration)

            elif type(obj) == Plane:
                dp = dot(obj.normal,(self.x-obj.x,self.y-obj.y))
                if dp < self.r: Collision(self,obj,[[self.x-obj.normal[0]*self.r,self.y-obj.normal[1]*self.r]],obj.normal,self.r-dp)


    def velocityAtPoint(self,p):
        return (self.xv - self.av * p[1], self.yv + self.av * p[0])

    def applyImpulse(self,impulse,contactVector):
        self.xv += impulse[0]*self.invMass
        self.yv += impulse[1]*self.invMass
        self.av += (contactVector[0]*impulse[1] - contactVector[1]*impulse[0])*self.invInertia

    def positionalCorrection(self,pv):
        self.x += pv[0]
        self.y += pv[1]

    def move(self):
        if self.invMass == 0: return
        self.xv += scene.gravity[0]
        self.yv += scene.gravity[1]

        self.xv += scene.wind[0]*self.invMass
        self.yv += scene.wind[1]*self.invMass
        
        self.x += self.xv
        self.y += self.yv
        self.a += self.av

        self.updateData()
        
class Polygon:
    def __init__(self,P,x,y,a=0,xv=0,yv=0,av=0,density=1,restitution=0.8,sf=0.5,df=0.5,immovable=False,colour=(255,255,255),layer=1):
        self.P = P
        self.c = com(self.P)
        self.P = [[point[0]-self.c[0],point[1]-self.c[1]] for point in self.P]
        self.x = x+self.c[0]
        self.y = y+self.c[1]
        self.a = math.radians(a)
        self.immovable = immovable
        self.xv = 0 if self.immovable else xv
        self.yv = 0 if self.immovable else yv
        self.av = 0 if self.immovable else math.radians(av)
        self.invMass = 0 if self.immovable else density/polyArea(self.P)
        self.invInertia = 0 if self.immovable else 1/(momentOfInertia(self.P)*density)
        self.restitution = restitution
        self.sf = sf
        self.df = df
        self.colour = colour

        self.layer = layer
        self.noCollisionGroup = float("nan")

        self.updateData()

        scene.objs.append(self)
        
    def draw(self):
        draw_polygon(screen,self.colour,self.dataAsPoints)
        #draw_rect(screen,(255,0,0),(self.AABB[0],self.AABB[1]),(self.AABB[2],self.AABB[3]))

    def updateData(self):
        self.dataAsPoints = dataToPoints(self.P,self.x,self.y,self.a)

        minX = float("inf")
        minY = float("inf")
        maxX = float("-inf")
        maxY = float("-inf")

        for x in self.dataAsPoints:
            if x[0] < minX: minX = x[0]
            if x[1] < minY: minY = x[1]
            if x[0] > maxX: maxX = x[0]
            if x[1] > maxY: maxY = x[1]
        
        self.AABB = (minX,maxY,maxX,minY)
            
    def collide(self,idx):
        A = self.dataAsPoints
        
        for i in range(idx+1,len(scene.objs)):
            obj = scene.objs[i]

            if self.invMass + obj.invMass == 0 or self.noCollisionGroup == obj.noCollisionGroup or self.layer & obj.layer == 0: continue

            if type(obj) == Circle:
                if not AABBAABBCollide(self.AABB,obj.AABB): continue
                
                collided,pvn,penetration,contacts = PolyCircleCollide(A,[obj.x,obj.y,obj.r])
                
                if collided:
                    Collision(self,obj,contacts,pvn,penetration)
                    
            elif type(obj) == Polygon:
                B = obj.dataAsPoints

                if not AABBAABBCollide(self.AABB,obj.AABB): continue
                
                collided,pv = PolyPolycollide(A,B,(self.x,self.y),(obj.x,obj.y))
                if collided:
                    magnitude = length(pv)
                    if magnitude == 0: continue
                    else: pvn = (pv[0]/magnitude,pv[1]/magnitude)

                    v1 = getFurthest(A,(-pvn[0],-pvn[1]))
                    v2 = getFurthest(B,pvn)
                    
                    lv1,lv2,edgel = getEdge(A,v1,0)
                    rv1,rv2,edger = getEdge(A,v1,1)
                    dl1 = abs(dot(edgel,pvn))
                    dr1 = abs(dot(edger,pvn))
                    if dl1<dr1:
                        edge1 = edgel
                        d1 = dl1
                        ev1 = (lv1,lv2)
                        e1 = v1
                    else:
                        edge1 = edger
                        d1 = dr1
                        ev1 = (rv1,rv2)
                        e1 = 3 if v1==0 else v1-1
                    lv1,lv2,edgel = getEdge(B,v2,0)
                    rv1,rv2,edger = getEdge(B,v2,1)
                    dl2 = abs(dot(edgel,pvn))
                    dr2 = abs(dot(edger,pvn))
                    if dl2<dr2:
                        edge2 = edgel
                        d2 = dl2
                        ev2 = (lv1,lv2)
                        e2 = v2
                    else:
                        edge2 = edger
                        d2 = dr2
                        ev2 = (rv1,rv2)
                        e2 = 3 if v2==0 else v2-1

                    if d1 < d2:
                        ref = edge1
                        inc = edge2
                        rv = ev1
                        iv = ev2
                    else:
                        ref = edge2
                        inc = edge1
                        rv = ev2
                        iv = ev1

                    ref = (-ref[0],-ref[1])
                    o1 = dot(ref,rv[0])
                    cp = clip(iv[0],iv[1],ref,o1)
                    if len(cp)<2: continue
                    o2 = dot(ref,rv[1])
                    cp = clip(cp[0],cp[1],(-ref[0],-ref[1]),-o2)
                    if len(cp)<2: continue
                    refNorm = (-ref[1],ref[0])

                    Collision(self,obj,[contact for contact in cp if dot(refNorm,(contact[0]-rv[0][0],contact[1]-rv[0][1]))>0],pvn,magnitude)

            elif type(obj) == Plane:
                minDP = float("inf")
                bestPoint = 0
                
                for point in A:
                    dp = dot(obj.normal,(point[0]-obj.x,point[1]-obj.y))
                    if dp < minDP:
                        minDP = dp
                        bestPoint = point

                if minDP < 0: Collision(self,obj,[bestPoint],obj.normal,abs(minDP))

    def velocityAtPoint(self,p):
        return (self.xv - self.av * p[1], self.yv + self.av * p[0])

    def applyImpulse(self,impulse,contactVector):
        self.xv += impulse[0]*self.invMass
        self.yv += impulse[1]*self.invMass
        self.av += (contactVector[0]*impulse[1] - contactVector[1]*impulse[0])*self.invInertia

    def positionalCorrection(self,pv):
        self.x += pv[0]
        self.y += pv[1]

    def move(self):
        if self.invMass == 0: return
        
        self.xv += scene.gravity[0]
        self.yv += scene.gravity[1]

        self.xv += scene.wind[0]*self.invMass
        self.yv += scene.wind[1]*self.invMass
        
        self.x += self.xv
        self.y += self.yv
        self.a += self.av

        self.updateData()

class Plane:
    def __init__(self,x,y,a,restitution=0.8,sf=0.5,df=0.5,colour=(255,255,255),layer=2047):
        self.x = x
        self.y = y
        self.a = math.radians(a)

        self.xv = 0
        self.yv = 0
        self.av = 0
        self.mass = 0
        self.invMass = 0
        self.invInertia = 0
        
        self.restitution = restitution
        self.sf = sf
        self.df = df
        self.colour = colour

        self.layer = layer
        self.noCollisionGroup = float("nan")

        self.cos = math.cos(self.a)
        self.sin = math.sin(self.a)
        self.tan = float("inf") if self.cos==0 else self.sin/self.cos
        self.cot = float("inf") if self.sin==0 else self.cos/self.sin

        self.normal = [-self.sin,self.cos]
        self.invNormal = [self.sin,-self.cos]

        scene.objs.append(self)

    def draw(self):
        pos = camera.toCameraSpace((self.x,self.y))

        points = []

        y1 = -pos[0]*self.tan+pos[1]
        if 0 <= y1 <= height: points.append((0,y1))

        y2 = (width-pos[0])*self.tan+pos[1]
        if 0 <= y2 <= height: points.append((width,y2))

        x1 = -pos[1]*self.cot+pos[0]
        if 0 <= x1 <= width: points.append((x1,0))

        x2 = (height-pos[1])*self.cot+pos[0]
        if 0 <= x1 <= width: points.append((x2,height))

        if len(points) >= 2: pygame.draw.aaline(screen,self.colour,points[0],points[1])

    def collide(self,idx):
        for i in range(idx+1,len(scene.objs)):
            obj = scene.objs[i]

            if obj.invMass == 0 or self.noCollisionGroup == obj.noCollisionGroup or self.layer & obj.layer == 0: continue

            if type(obj) == Circle:
                dp = dot(self.normal,(obj.x-self.x,obj.y-self.y))
                if dp < obj.r: Collision(self,obj,[[obj.x-self.normal[0]*obj.r,obj.y-self.normal[1]*obj.r]],self.invNormal,obj.r-dp)
                
            elif type(obj) == Polygon:
                B = obj.dataAsPoints

                minDP = float("inf")
                bestPoint = 0
                
                for point in B:
                    dp = dot(self.normal,(point[0]-self.x,point[1]-self.y))
                    if dp < minDP:
                        minDP = dp
                        bestPoint = point

                if minDP < 0: Collision(self,obj,[bestPoint],self.invNormal,abs(minDP))

    def velocityAtPoint(self,p):
        return 0

    def applyImpulse(self,impulse,contactVector):
        return

    def positionalCorrection(self,pv):
        return

    def move(self):
        return

class Spring:
    def __init__(self,A,B,p1,p2,s,d,l=None):
        self.A = A
        self.B = B
        self.p1 = p1
        self.p2 = p2
        self.s = s
        self.d = d
        if l == None:
            p1 = rotate(self.p1,self.A.a)
            p2 = rotate(self.p2,self.B.a)
            l = dist((p1[0]+self.A.x,p1[1]+self.A.y),(p2[0]+self.B.x,p2[1]+self.B.y))
            self.l = l
        else:
            self.l = l

        scene.springs.append(self)

    def draw(self,col=(255,255,255)):
        p1 = rotate(self.p1,self.A.a)
        p2 = rotate(self.p2,self.B.a)
        p1 = camera.toCameraSpace((p1[0]+self.A.x,p1[1]+self.A.y))
        p2 = camera.toCameraSpace((p2[0]+self.B.x,p2[1]+self.B.y))
        p1 = (int(p1[0]),int(p1[1]))
        p2 = (int(p2[0]),int(p2[1]))
        pygame.draw.line(screen,col,p1,p2,max(1,int(camera.m*4)))
        pygame.draw.circle(screen,col,p1,max(1,int(camera.m*5)))
        pygame.draw.circle(screen,col,p2,max(1,int(camera.m*5)))

    def update(self):
        rA = rotate(self.p1,self.A.a)
        rB = rotate(self.p2,self.B.a)
        p1 = (rA[0]+self.A.x,rA[1]+self.A.y)
        p2 = (rB[0]+self.B.x,rB[1]+self.B.y)

        vec = (p2[0]-p1[0],p2[1]-p1[1])
        mag = length(vec)
        if mag == 0: return
        vec = (vec[0]/mag,vec[1]/mag)
        
        vA = self.A.velocityAtPoint(self.p1)
        vB = self.B.velocityAtPoint(self.p2)
        
        F = self.s * (mag - self.l) + dot((vB[0]-vA[0],vB[1]-vA[1]),vec)*self.d
        impulse = (vec[0]*F,vec[1]*F)

        self.A.applyImpulse(impulse,rA)
        self.B.applyImpulse((-impulse[0],-impulse[1]),rB)


#JANKY
class Fixjoint:
    def __init__(self,A,B,p1,p2):
        self.A = A
        self.B = B
        self.p1 = p1
        self.p2 = p2

        self.invMassSum = self.A.invMass + self.B.invMass
        if self.invMassSum != 0:
            self.Acorrect = self.A.invMass / self.invMassSum
            self.Bcorrect = self.B.invMass / self.invMassSum

        self.invInertiaSum = self.A.invInertia + self.B.invInertia
        if self.invMassSum != 0:
            self.AinvInertiaPortion = self.A.invInertia / self.invInertiaSum
            self.BinvInertiaPortion = self.B.invInertia / self.invInertiaSum

        a = type(self.A.noCollisionGroup) == NoCollisionGroup
        b = type(self.B.noCollisionGroup) == NoCollisionGroup

        if a:
            if b:
                self.A.noCollisionGroup.merge(self.B.noCollisionGroup)
                self.noCollisionGroup = self.A.noCollisionGroup
            else:
                self.A.noCollisionGroup.add(self.B)
                self.noCollsionGroup = self.A.noCollisionGroup
        else:
            if b:
                self.B.noCollisionGroup.add(self.A)
                self.noCollsionGroup = self.B.noCollisionGroup
            else:
                self.noCollsionGroup = NoCollisionGroup(self.A,self.B)

        self.ogA = self.A.a
        self.relA = self.A.a-self.B.a

        scene.fixjoints.append(self)

    def draw(self):
        pos = rotate(self.p1,self.A.a)
        draw_circle_with_cross(screen,(0,255,0),10,pos[0]+self.A.x,pos[1]+self.A.y,self.A.a-self.ogA)

    def update(self,div=4):
        avg = ((self.A.xv+self.B.xv)/2,(self.A.yv+self.B.yv)/2)
        self.A.xv = avg[0]
        self.B.xv = avg[0]
        self.A.yv = avg[1]
        self.B.yv = avg[1]

        s = self.A.av+self.B.av
        self.A.av = s*self.AinvInertiaPortion
        self.B.av = s*self.BinvInertiaPortion
        
        rA = rotate(self.p1,self.A.a)
        rB = rotate(self.p2,self.B.a)
        p1 = (rA[0]+self.A.x,rA[1]+self.A.y)
        p2 = (rB[0]+self.B.x,rB[1]+self.B.y)

        vec = ((p2[0]-p1[0]),(p2[1]-p1[1]))

        relA = self.relA - (self.A.a-self.B.a)

        if self.A.invMass != 0:
            self.A.applyImpulse((vec[0]/div/self.A.invMass,vec[1]/div/self.A.invMass),rA)
            correctionA = (vec[0]*self.Acorrect,vec[1]*self.Acorrect)
            self.A.positionalCorrection(correctionA)
            self.A.a += relA*self.Acorrect
            
        if self.B.invMass != 0:
            self.B.applyImpulse((-vec[0]/div/self.B.invMass,-vec[1]/div/self.B.invMass),rB)
            correctionB = (-vec[0]*self.Bcorrect,-vec[1]*self.Bcorrect)
            self.B.positionalCorrection(correctionB)
            self.B.a -= relA*self.Bcorrect

class Hinge:
    def __init__(self,A,B,p1,p2,l=None):
        self.A = A
        self.B = B
        self.p1 = p1
        self.p2 = p2
        if l == None:
            p1 = rotate(self.p1,self.A.a)
            p2 = rotate(self.p2,self.B.a)
            l = dist((p1[0]+self.A.x,p1[1]+self.A.y),(p2[0]+self.B.x,p2[1]+self.B.y))
            self.l = l
        else:
            self.l = l

        self.invMassSum = self.A.invMass + self.B.invMass
        if self.invMassSum != 0:
            self.Acorrect = self.A.invMass / self.invMassSum
            self.Bcorrect = self.invMassSum - self.Acorrect

        scene.hinges.append(self)

    def draw(self,col=(255,127,0)):
        p1 = rotate(self.p1,self.A.a)
        p2 = rotate(self.p2,self.B.a)
        p1 = camera.toCameraSpace((p1[0]+self.A.x,p1[1]+self.A.y))
        p2 = camera.toCameraSpace((p2[0]+self.B.x,p2[1]+self.B.y))
        p1 = (int(p1[0]),int(p1[1]))
        p2 = (int(p2[0]),int(p2[1]))
        pygame.draw.line(screen,col,p1,p2,max(1,int(camera.m*4)))
        pygame.draw.circle(screen,col,p1,max(1,int(camera.m*5)))
        pygame.draw.circle(screen,col,p2,max(1,int(camera.m*5)))

    def update(self,percentage=0.25):
        rA = rotate(self.p1,self.A.a+self.A.av)
        rB = rotate(self.p2,self.B.a+self.B.av)
        p1 = (rA[0]+self.A.x+self.A.xv,rA[1]+self.A.y+self.A.yv)
        p2 = (rB[0]+self.B.x+self.B.xv,rB[1]+self.B.y+self.B.yv)

        vec = ((p2[0]-p1[0]),(p2[1]-p1[1]))
        mag = length(vec)
        if mag == 0: return
        vecn = (vec[0]/mag,vec[1]/mag)

        vec = ((vec[0]-vecn[0]*self.l)*percentage,(vec[1]-vecn[1]*self.l)*percentage)
        
        if self.A.invMass != 0:
            self.A.applyImpulse((vec[0]/self.A.invMass,vec[1]/self.A.invMass),rA)
            correctionA = (vec[0]*self.Acorrect,vec[1]*self.Acorrect)
            self.A.positionalCorrection(correctionA)
            
        if self.B.invMass != 0:
            self.B.applyImpulse((-vec[0]/self.B.invMass,-vec[1]/self.B.invMass),rB)
            correctionB = (-vec[0]*self.Bcorrect,-vec[1]*self.Bcorrect)
            self.B.positionalCorrection(correctionB)

class Axle:
    def __init__(self,A,B,p1,p2,rpm=0,torque=0):
        self.A = A
        self.B = B
        self.p1 = p1
        self.p2 = p2
        self.rpm = rpm
        self.torque = torque

        self.invMassSum = self.A.invMass + self.B.invMass
        if self.invMassSum != 0:
            self.Acorrect = self.A.invMass / self.invMassSum
            self.Bcorrect = self.invMassSum - self.Acorrect

        a = type(self.A.noCollisionGroup) == NoCollisionGroup
        b = type(self.B.noCollisionGroup) == NoCollisionGroup

        if a:
            if b:
                self.A.noCollisionGroup.merge(self.B.noCollisionGroup)
                self.noCollisionGroup = self.A.noCollisionGroup
            else:
                self.A.noCollisionGroup.add(self.B)
                self.noCollsionGroup = self.A.noCollisionGroup
        else:
            if b:
                self.B.noCollisionGroup.add(self.A)
                self.noCollsionGroup = self.B.noCollisionGroup
            else:
                self.noCollsionGroup = NoCollisionGroup(self.A,self.B)

        self.ogSum = self.A.a+self.B.a

        scene.axles.append(self)

    def draw(self):
        r = 1
        
        pos = rotate(self.p1,self.A.a)
        draw_circle(screen,(0,0,255),pos[0]+self.A.x,pos[1]+self.A.y,r*10)
        draw_circle(screen,(0,0,255),pos[0]+self.A.x,pos[1]+self.A.y,r*4)

        angle = self.A.a+self.B.a - self.ogSum

        for i in range(8):
            draw_circle(screen,(0,0,255),pos[0]+self.A.x+math.cos(angle)*r*7,pos[1]+self.A.y+math.sin(angle)*r*7,r)
            angle += math.pi/4

    def update(self,percentage=0.25):
        rA = rotate(self.p1,self.A.a+self.A.av)
        rB = rotate(self.p2,self.B.a+self.B.av)
        p1 = (rA[0]+self.A.x+self.A.xv,rA[1]+self.A.y+self.A.yv)
        p2 = (rB[0]+self.B.x+self.B.xv,rB[1]+self.B.y+self.B.yv)

        vec = ((p2[0]-p1[0])*percentage,(p2[1]-p1[1])*percentage)
        
        if self.A.invMass != 0:
            self.A.applyImpulse((vec[0]/self.A.invMass,vec[1]/self.A.invMass),rA)
            correctionA = (vec[0]*self.Acorrect,vec[1]*self.Acorrect)
            self.A.positionalCorrection(correctionA)
            
        if self.B.invMass != 0:
            self.B.applyImpulse((-vec[0]/self.B.invMass,-vec[1]/self.B.invMass),rB)
            correctionB = (-vec[0]*self.Bcorrect,-vec[1]*self.Bcorrect)
            self.B.positionalCorrection(correctionB)

thrusterPoly = [[20,-20],[20,-30],[50,0],[20,30],[20,20],[-30,20],[-30,10],[-50,20],[-50,-20],[-30,-10],[-30,-20]]

class Thruster:
    def __init__(self,A,p,a,F,orients=True):
        self.A = A
        self.p = p
        self.a = math.radians(a)
        self.F = F
        self.orients = orients

        self.pos = rotate(self.p,self.A.a)
        if self.orients: self.angle = self.A.a + self.a
        else: self.angle = self.a

        scene.thrusters.append(self)

    def draw(self):
        draw_polygon(screen,(255,0,0),dataToPoints(thrusterPoly,self.pos[0]+self.A.x,self.pos[1]+self.A.y,self.angle))

    def update(self):
        self.pos = rotate(self.p,self.A.a)
        if self.orients: self.angle = self.A.a + self.a
        else: self.angle = self.a
        
        self.A.applyImpulse([math.cos(self.angle)*self.F,math.sin(self.angle)*self.F],self.pos)

class Tracer:
    def __init__(self,A,p,r,fade=100,colour=(255,0,255)):
        self.A = A
        self.p = p
        self.r = r
        self.fade = fade
        self.colour = colour

        scene.tracers.append(self)

    def draw(self):
        pos = rotate(self.p,self.A.a)
        draw_circle(screen,self.colour,pos[0]+self.A.x,pos[1]+self.A.y,self.r)
        draw_circle(screen,self.colour,pos[0]+self.A.x,pos[1]+self.A.y,self.r*0.8)

    def update(self):
        pass

class Laser:
    def __init__(self,A,p,a,w,maxDist,colour):
        self.A = A
        self.p = p
        self.a = math.radians(a)
        self.w = w
        self.maxDist = maxDist
        self.colour = colour

        scene.lasers.append(self)

    def draw(self):
        pass

    def update(self):
        pass

pygame.init()

info = pygame.display.Info()
width = info.current_w
height = info.current_h
hwidth = width/2
hheight = height/2

screen = pygame.display.set_mode((width,height),pygame.FULLSCREEN)

clock = pygame.time.Clock()

bg = (0,0,0)

FPS = 120

camera = Camera()
scene = Scene()

#COUNTERCLOCKWISE DEFINITION FOR POLYGONS
# NO CONCAVITIES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# (circles collide with concave angles correctly but don't resolve properly)
# (polygons with concave angles just don't work)

Plane(0,0,0)
Plane(-10000,0,-90)
Plane(10000,0,90)
Plane(0,10000,180)

Polygon([(0,0),(3000,0),(0,1000)],0,0,0,immovable=True)

##Circle(100,-1000,1000,0)
##Circle(100,-1400,1000,0)
##Polygon([(-1420,1020),(-1420,980),(-980,980),(-980,1020)],0,0,0)
##Axle(scene.objs[-3],scene.objs[-1],(0,0),(-1000-scene.objs[-1].x,1000-scene.objs[-1].y))
##Axle(scene.objs[-2],scene.objs[-1],(0,0),(-1400-scene.objs[-1].x,1000-scene.objs[-1].y))

Circle(100,-1000,1000,0)
Circle(100,-1400,1000,0)
Hinge(scene.objs[-2],scene.objs[-1],(0,50),(0,-50))

brush = 0
maxBrush = 3
dragging = 0
startDragPos = None
    
while 1:
    dt = clock.tick(FPS)

    screen.fill(bg)

    mx,my = pygame.mouse.get_pos()
    worldPoint = camera.toWorldSpace((mx,my))

    keys = pygame.key.get_pressed()
    shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
    ctrl = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]

    left,middle,right,up,down = pygame.mouse.get_pressed(5)

    if left:
        if dragging:
            if brush < 2:
                draw_line(screen,(255,0,0),startDragPos,worldPoint)
            elif brush == 2:
                draw_rect(screen,(255,0,0),startDragPos,worldPoint)
            elif brush == 3:
                draw_circle(screen,(255,0,0),startDragPos[0],startDragPos[1],dist(startDragPos,worldPoint))
        else:
            dragging = 1
            startDragPos = worldPoint
    else:
        if dragging:
            dragging = 0
            if brush == 0:
                P = generateConvex(random.randint(25,150),random.randint(3,10))
                pos = startDragPos
                vel = ((startDragPos[0]-worldPoint[0])/10,(startDragPos[1]-worldPoint[1])/10)
                
                Polygon(P,pos[0],pos[1],0,vel[0],vel[1],0,immovable=ctrl)
            elif brush == 1:
                pos = startDragPos
                vel = ((startDragPos[0]-worldPoint[0])/10,(startDragPos[1]-worldPoint[1])/10)
                Circle(random.randint(5,50),pos[0],pos[1],0,vel[0],vel[1],0,immovable=ctrl)

            elif brush == 2:
                p1 = startDragPos
                p2 = worldPoint

                if p1[0] == p2[0] or p1[1] == p2[1]: continue

                if p1[0] > p2[0]:
                    p1[0],p2[0] = (p2[0],p1[0])

                if p1[1] < p2[1]:
                    p1[1],p2[1] = (p2[1],p1[1])
                
                Polygon([p1,(p1[0],p2[1]),p2,(p2[0],p1[1])],0,0,0,immovable=ctrl)

            elif brush == 3:
                Circle(dist(startDragPos,worldPoint),startDragPos[0],startDragPos[1],0,immovable=ctrl)

    scrolling = 0
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_SPACE:
                scene.paused = not scene.paused
            elif event.key == pygame.K_TAB:
                brush += 1
                if brush > maxBrush: brush = 0

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: scrolling = 1
            elif event.button == 5: scrolling = -1

    scene.update()
    camera.update(keys,scrolling,shift,dt)
    scene.draw()
    
    pygame.display.update()
