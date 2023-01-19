class test:
    def __init__(self,objs):
        self.enabled = 1
        for obj in objs:
            obj.group = self

class obj:
    def __init__(self):
        self.group = None

    def getGroup(self):
        if self.group.enabled:
            return id(self.group)
        else:
            self.group = None
            return float("nan")


obj1 = obj()
obj2 = obj()

test1 = test([obj1,obj2])

print(obj1.getGroup())
print(obj2.getGroup())

test1.enabled = 0
del test1

print(obj1.getGroup())
print(obj2.getGroup())
