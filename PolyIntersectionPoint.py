def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

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

    _1mix = min(x1)
    _1max = max(x1)
    _2mix = min(x2)
    _2max = max(x2)
    _1miy = min(y1)
    _1may = max(y1)
    _2miy = min(y2)
    _2may = max(y2)

    if (x>_1mix)and(x>_2mix)and(x<_1max)and(x<_2max)and(y>_1miy)and(y>_2miy)and(y<_1may)and(y<_2may):    
        return True,[x, y]
    else:
        return False,None

def getContact(pointsA,pointsB):
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


    return intersections

print(getContact([[0,0],[1,1],[-1,2]],[[5,5],[0,1],[2,-4]]))
