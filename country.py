class Country:

    listMaxCity = {}

    def __init__(self, name, continent, population, area, cityPopul, city):
        self.name = name
        self.continent = continent
        self.population = population
        self.area = area
        self.cityPopul = cityPopul
        self.city = city

    def printInformation (self):
        print("Інформація про введені дані:")
        print("Name country: ", self.name)
        print("Continent: ",self.continent)
        print("Area: ", self.area)
        print("Millioner city: ", self.city)
        print(' ')

    def densityPop (self):
        return self.population / self.area;

    def procCity (self):
        return (self.cityPopul * 100) / self.population

    def printDensity (self):
        print("Density of population: ", self.densityPop())
        print(' ')

    def printProcent (self):
        print("Procent of cityPopul: ", self.procCity())
        print(' ')

    def maxCity (self):
        result1 = ' '
        result2 = 0

        for keyi,vali in self.city.items():
            if (vali > result2):
                result2 = vali
                result1 = keyi
        self.listMaxCity[result1] = result2
    def findMaxCity (self):
        self.maxCity ()
        result = ['', 0]
        for keyi, vali in self.listMaxCity.items():
            if (vali > result[1]):
                result[1] = vali
                result[0] = keyi
        print('The biggest city is ', result[0], ", its population = ", result[1])



c1 = Country('Ukraina', 'Europa', 42234014, 603628, 69.2, {'Kyiv': 350943})
c2 = Country('The United States of America', 'North America', 1001559000, 42549000, 81.6, {'New York City': 8175133, 'Los Angeles': 3792621, 'Chicago': 2695598})
c3 = Country('Russia', 'Europa/Asia',146880432, 17125191, 74.0, {'Moskow': 12506468, 'Sankt-peterburg': 5351935})

c1.printInformation()
c2.printInformation()
c3.printInformation()

c1.printDensity()
c1.printProcent()

c2.printDensity()
c2.printProcent()

c3.printDensity()
c3.printProcent()

c1.maxCity()
c2.maxCity()
c3.maxCity()

c1.findMaxCity()