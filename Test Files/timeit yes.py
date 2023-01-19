import timeit

setup1 = """

"""

code1 = """
a = None
b = (a is None)

"""

setup2 = """

"""

code2 = """
a = None
b = a == None

"""

print(timeit.timeit(code1,setup1))
print(timeit.timeit(code2,setup2))
