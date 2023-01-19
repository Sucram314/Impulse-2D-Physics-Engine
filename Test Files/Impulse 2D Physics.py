import pygame
from pygame import gfxdraw
import math
import random

pygame.init()

a = pygame.image.load("C:\\Users\\marcu\\OneDrive\\Pictures\\Saved Pictures\\unnamed.jpg")
pygame.display.set_icon(a)
pygame.display.set_caption("2D Physics")

clock = pygame.time.Clock()

bg = (0,0,0)

pygame.init()

width = 1280
height = 658

FPS = 240

screen = pygame.display.set_mode([width,height])

a = True
b = False
randColour = (255,255,255)

camera = [0,0,1]

mx,my = (0,0)

mxo,myo = (0,0)

gravity = 0.02
slop = 0.01
percent = 0.6
recalculations = 0

mode = 0

def dotproduct(p1,p2):
    return p1[0]*p2[0]+p1[1]*p2[1]


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

#circle: type 0
class circle:
    def __init__(self,centrex,centrey,radius,xVel,yVel,density,restitution,friction,colour):
        self.centrex = centrex
        self.centrey = centrey
        self.radius = radius
        self.xVel = xVel
        self.yVel = yVel
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

    def update(self):
        if self.invMass()!=0:
            self.yVel -= gravity
        self.centrex += self.xVel
        self.centrey += self.yVel

    def detect(self):
        collisions = []
        for thing in things:
            if thing!=self:
                if thing.type()==0:
                    d = (thing.centrex-self.centrex)**2+(thing.centrey-self.centrey)**2
                    if d<((self.radius+thing.radius)**2):
                        d = d**0.5
                        if d!=0:
                            normal = [(thing.centrex-self.centrex)/d,(thing.centrey-self.centrey)/d]
                            penetration = self.radius + thing.radius - d
                        else:
                            normal = [1,0]
                            penetration = self.radius
                        collisions.append([thing,normal,penetration])

                if thing.type()==1:
                    inside = False
                    
                    cX = clamp(self.centrex,thing.left,thing.right)
                    cY = clamp(self.centrey,thing.bottom,thing.top)

                    if (cX==self.centrex)and(cY==self.centrey):
                        inside = True
                        nlX = abs(self.centrex - thing.left)
                        nrX = abs(self.centrex - thing.right)
                        ntY = abs(self.centrey - thing.top)
                        nbY = abs(self.centrey - thing.bottom)

                        least = min(nlX,nrX,ntY,nbY)

                        if least == nlX:
                            cX = thing.left
                            cY = self.centrey
                        elif least == nrX:
                            cX = thing.right
                            cY = self.centrey
                        elif least == ntY:
                            cX = self.centrex
                            cY = thing.top
                        elif least == nbY:
                            cX = self.centrex
                            cY = thing.bottom

                    dX = self.centrex - cX
                    dY = self.centrey - cY

                    d = (dX * dX) + (dY * dY)
                    
                    if not(d>(self.radius*self.radius))or inside:
                        d = d**0.5
                        normal = [(cX-self.centrex),(cY-self.centrey)]
                        magnit = (normal[0]**2+normal[1]**2)**0.5
                        normal = [normal[0]/magnit,normal[1]/magnit]
                        penetration = self.radius - d
                        if inside:
                            normal = [normal[0]*-1,normal[1]*-1]
                            penetration += 2*d
                        
                        collisions.append([thing,normal,penetration])
        
        return collisions

    def resolve(self,collisions):
        for collision in collisions:
            rv = [collision[0].xVel-self.xVel,collision[0].yVel-self.yVel]
            velAlongNormal = dotproduct(rv,collision[1])

            if velAlongNormal <= 0:
                e = min(collision[0].restitution,self.restitution)

                j = -(1+e) * velAlongNormal
                j /= (self.invMass()+collision[0].invMass())

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
                    jt /= (self.invMass()+collision[0].invMass())

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

                correction = (max(collision[2]-slop,0)/(self.invMass()+collision[0].invMass()))*percent
                correction = [collision[1][0]*correction,collision[1][1]*correction]

                self.centrex -= self.invMass() * correction[0]
                self.centrey -= self.invMass() * correction[1]

                if collision[0].type() == 0:
                    collision[0].centrex += collision[0].invMass() * correction[0]
                    collision[0].centrey += collision[0].invMass() * correction[1]
                elif collision[0].type() == 1:
                    xCorrect = collision[0].invMass()*correction[0]
                    yCorrect = collision[0].invMass()*correction[1]
                    collision[0].left += xCorrect
                    collision[0].right += xCorrect
                    collision[0].top += yCorrect
                    collision[0].bottom += yCorrect
            

    def draw(self):
        pygame.gfxdraw.aacircle(screen,int((self.centrex+camera[0])/camera[2]+(width/2)),int((-self.centrey+camera[1])/camera[2]+(height/2)),int(self.radius/camera[2]),self.colour)

#AABB: type 1
class AABB:
    def __init__(self,left,top,right,bottom,xVel,yVel,density,restitution,friction,colour):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.xVel = xVel
        self.yVel = yVel
        self.density = density
        self.restitution = restitution
        self.friction = friction
        self.colour = colour

    def type(self):
        return 1

    def invMass(self):
        if self.density==0:
            return 0
        volume = (abs(self.right-self.left))*(abs(self.top-self.bottom))
        return 1/((self.density/100)*volume)

    def update(self):
        if self.invMass()!=0:
            self.yVel -= gravity
        self.left += self.xVel
        self.top += self.yVel
        self.right += self.xVel
        self.bottom += self.yVel

    def detect(self):
        collisions = []
        
        for thing in things:
            if thing!=self:
                if thing.type()==0:
                    inside = False
                    
                    cX = clamp(thing.centrex,self.left,self.right)
                    cY = clamp(thing.centrey,self.bottom,self.top)

                    if (cX==thing.centrex)and(cY==thing.centrey):
                        inside = True
                        nlX = abs(thing.centrex - self.left)
                        nrX = abs(thing.centrex - self.right)
                        ntY = abs(thing.centrey - self.top)
                        nbY = abs(thing.centrey - self.bottom)

                        least = min(nlX,nrX,ntY,nbY)

                        if least == nlX:
                            cX = self.left
                            cY = thing.centrey
                        elif least == nrX:
                            cX = self.right
                            cY = thing.centrey
                        elif least == ntY:
                            cX = thing.centrex
                            cY = self.top
                        elif least == nbY:
                            cX = thing.centrex
                            cY = self.bottom

                    dX = thing.centrex - cX
                    dY = thing.centrey - cY

                    d = (dX * dX) + (dY * dY)
                    
                    if not(d>(thing.radius*thing.radius))or inside:
                        d = d**0.5
                        normal = [(cX-thing.centrex),(cY-thing.centrey)]
                        magnit = (normal[0]**2+normal[1]**2)**0.5
                        normal = [normal[0]/-magnit,normal[1]/-magnit]
                        penetration = thing.radius - d
                        if inside:
                            normal = [normal[0]*-1,normal[1]*-1]
                            penetration += 2*d
                        collisions.append([thing,normal,penetration])

                if thing.type()==1:
                    n = [((thing.left+thing.right)/2)-((self.left+self.right)/2),((thing.top+thing.bottom)/2)-((self.top+self.bottom)/2)]

                    a_extent = (self.right-self.left)/2
                    b_extent = (thing.right-thing.left)/2

                    x_overlap = a_extent + b_extent - abs(n[0])

                    if (x_overlap > 0):
                        a_extent = (self.top-self.bottom)/2
                        b_extent = (thing.top-thing.bottom)/2

                        y_overlap = a_extent + b_extent - abs(n[1])

                        if (y_overlap > 0):
                            if (x_overlap < y_overlap):
                                if (n[0] < 0):
                                    normal = [-1,0]
                                else:
                                    normal = [1,0]
                                penetration = x_overlap
                            else:
                                if (n[1] < 0):
                                    normal = [0,-1]
                                else:
                                    normal = [0,1]
                                penetration = y_overlap

                            collisions.append([thing,normal,penetration])
        
        return collisions

    def resolve(self,collisions):
        for collision in collisions:
            rv = [collision[0].xVel-self.xVel,collision[0].yVel-self.yVel]
            velAlongNormal = dotproduct(rv,collision[1])

            if velAlongNormal <= 0:
                e = min(collision[0].restitution,self.restitution)

                j = -(1+e) * (velAlongNormal)
                j /= (self.invMass()+collision[0].invMass())

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
                    jt /= (self.invMass()+collision[0].invMass())

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

                correction = (max(collision[2]-slop,0)/(self.invMass()+collision[0].invMass()))*percent
                correction = [collision[1][0]*correction,collision[1][1]*correction]

                xCorrect = self.invMass()*correction[0]
                yCorrect = self.invMass()*correction[1]
                self.left -= xCorrect
                self.right -= xCorrect
                self.top -= yCorrect
                self.bottom -= yCorrect

                if collision[0].type() == 0:
                    collision[0].centrex += collision[0].invMass() * correction[0]
                    collision[0].centrey += collision[0].invMass() * correction[1]
                elif collision[0].type() == 1:
                    xCorrect = collision[0].invMass()*correction[0]
                    yCorrect = collision[0].invMass()*correction[1]
                    collision[0].left += xCorrect
                    collision[0].right += xCorrect
                    collision[0].top += yCorrect
                    collision[0].bottom += yCorrect

    def draw(self):
        pygame.gfxdraw.rectangle(screen,(int((self.left+camera[0])/camera[2]+(width/2)),int((-self.top+camera[1])/camera[2]+(height/2)),int((self.right-self.left)/camera[2]),int((self.top-self.bottom)/camera[2])),self.colour)

things = [AABB(-9999999,0,9999999,-9999999,0,0,0,0.7,0.2,(255,255,255)),circle(0,1000,500,0,0,0,0.7,0.2,(255,255,255))]

ogLen = len(things)

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
                mode = 1-mode
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
            things.append(circle((mx-(width/2))*camera[2]-camera[0]+random.randint(-1000,1000),(my-(height/2))*-camera[2]+camera[1]+random.randint(-1000,1000),random.randint(10,200),0,0,0.2,0.7,0.2,(random.randint(150,255),random.randint(150,255),random.randint(150,255))))
        elif mode == 1:
            randLength = random.randint(50,200)
            randHeight = random.randint(50,200)
            randOffx = random.randint(-1000,1000)
            randOffy = random.randint(-1000,1000)
            things.append(AABB((mx-(width/2))*camera[2]-camera[0]-randLength/2+randOffx,(my-(height/2))*-camera[2]+camera[1]+randHeight/2+randOffy,(mx-(width/2))*camera[2]-camera[0]+randLength/2+randOffx,(my-(height/2))*-camera[2]+camera[1]-randHeight/2+randOffy,0,0,0.2,0.7,0.2,(random.randint(150,255),random.randint(150,255),random.randint(150,255))))

    if keys[pygame.K_r]:
        things = things[:ogLen]

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
                things.append(circle((mxo-(width/2))*camera[2]-camera[0],(myo-(height/2))*-camera[2]+camera[1],random.randint(10,200),(mxo*camera[2]-mx*camera[2])/100,(-myo*camera[2]+my*camera[2])/100,0.2,0.7,0.2,randColour))
            elif randObject == 1:
                randLength = random.randint(50,200)
                randHeight = random.randint(50,200)
                things.append(AABB((mxo-(width/2))*camera[2]-camera[0]-randLength/2,(myo-(height/2))*-camera[2]+camera[1]+randHeight/2,(mxo-(width/2))*camera[2]-camera[0]+randLength/2,(myo-(height/2))*-camera[2]+camera[1]-randHeight/2,(mxo*camera[2]-mx*camera[2])/100,(-myo*camera[2]+my*camera[2])/100,0.2,0.7,0.2,randColour))

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
