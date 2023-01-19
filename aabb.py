def AABBAABBCollide(A,B):
    return A[0] <= B[2] and B[0] <= A[2] and A[3] <= B[1] and B[3] <= A[1]


print(AABBAABBCollide((0,5,5,0),(-5,4,-1,0)))
