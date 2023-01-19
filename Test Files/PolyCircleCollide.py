import pygame
from pygame import gfxdraw

pygame.init()

screen = pygame.display.set_mode([1280,658])

def dotproduct(v1,v2):
    return v1[0]*v2[0]+v1[1]*v2[1]

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
    nextIndex = 0
    for i in range(len(A)):
        nextIndex = i+1
        if nextIndex == len(A):
            nextIndex = 0

        vc = A[i]
        vn = A[nextIndex]

        collided,normal,penetration,radiusa,radiusb = edgeCirclecollide(B,vc,vn)

        if collided:
            radiusb = [radiusb[0],radiusb[1]]
            return True,normal,penetration,radiusa,radiusb

    inside,normal,penetration,radiusa,radiusb = polygonPoint(A,[B[0],B[1]],B[2])
    if inside:
        return True,normal,penetration,radiusa,radiusb

    return False,None,None,None,None

clock = pygame.time.Clock()

poly1 = [[1.2328734000040455,121.93996050845794],[-72.32057714114805,68.68935141290797],[-90.10602382767252,53.81614161096047],[-140.29685858013173,11.09464267899071],[-136.51549250644737,-29.963645284959256],[-135.05932309573217,-38.54432493634735],[-119.99426354878833,-94.28183057617187],[-84.43587159751299,-131.11111545225947],[-55.216805003443326,-137.41827061963994],[2.5109752775536407,-136.09747705462019],[27.12148179690716,-124.87632687155161],[72.52366035718657,-81.02342016411353],[115.21749267530856,-24.508937019169252],[134.07141736419987,2.023597289974532],[140.29685858013175,25.289606798596424],[136.6493410957932,47.4886936660142],[114.23647318121601,70.25504308532322],[53.641981875152766,127.9863360571521],[51.26023331891234,130.04025820597283],[24.125059708344445,137.41827061963994]]

circle = [0,0,20]

running = True
while running:
    mx,my = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0,0,0))

    poly1draw = []
    for point in poly1:
        poly1draw.append([point[0]+640,-point[1]+329])

    circle = [mx,my,20]

    pygame.gfxdraw.aacircle(screen,circle[0],circle[1],circle[2],(255,255,255))

    collided,a,b,c,d = PolyCirclecollide(poly1draw,circle)
    if collided:
        c = [c[0],c[1]]
        d = [d[0],d[1]]
        
        pygame.draw.circle(screen,(0,255,0),tuple(c),10)
        pygame.draw.circle(screen,(0,0,255),tuple(d),10)
        pygame.gfxdraw.aapolygon(screen,[tuple(x) for x in poly1draw],(255,0,0))
    else:
        pygame.gfxdraw.aapolygon(screen,[tuple(x) for x in poly1draw],(255,255,255))

    pygame.display.update()

    clock.tick(120)

pygame.quit()
