import math

class Circule:

    circles = []

    def __init__(self,r,x,y):
        self.radius = r
        self.x = x
        self.y = y
        self.circles.append([r,x,y])

    def __str__(self):
        return 'Коло радіусом {} з центром в точці ({} , {})'.format(self.radius, self.x, self.y)

    def areaCircle (self):
        return math.pi + math.pow(self.radius, 2)

    def diametrCirlce (self):
        return 2 * self.radius

    def lengthCirle (self):
        return 2 * math.pi * self.radius

    def areaSector (self):
        return self.lengthCirle() * self.radius * 0.5

    def arcLength (self, angel):
        return (math.pi * self.radius * angel) / 180

    def intersecCir (self, r, a, b):
        try:
            distance = math.sqrt(math.pow(self.x - a, 2) + math.pow(self.y - b, 2))
            if (distance == self.radius + r):
                print("Два кола мають зовнішній дотик")
            elif (distance < self.radius + r):
                print("Два кола перетинаються")
            else:
                print("Кола не перетинаються")
        except:
            print("Введені не коректні дані")




c = Circule(1, 2, 3)
print(c)
print("Площа кола рівна: ", c.areaCircle())
print("Радіус кола рівна: ", c.diametrCirlce())
print("Довжина кола рівна: ", c.lengthCirle())
print("Площа сектора, яка задається колом рівна: ", c.areaSector())
print("Довжина дуги кола рівна: ", c.arcLength(130))
c.intersecCir(2, 13, 5)

v = Circule(3, 6, 9)
print(v)

c.intersecCir(v.radius, v.x, v.y)
