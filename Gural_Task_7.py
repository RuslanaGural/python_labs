#Створення прикладних програм на мові Python. Лабораторна робота №6.2. Гураль Руслана. FI-9119

print('''' Створення прикладних програм на мові Python. Лабораторна робота №6.2
  Гураль Руслана. FI-9119''')
import datetime

def initPupils(list):
    surname = input('Surname: ')
    firstname = input('Firstname: ')
    patronymic = input('Patronymic: ')
    dateOfBirth = input('Date of birth (in format year-month-number): ')

    list.append({'surname': surname, 'firstname': firstname, 'patronymic': patronymic, 'dateOfBirth': dateOfBirth})
    return list

def printValues(list):
    for i in range(0,len(list)):
        j = list[i]
        print(j.get('surname'), j.get('firstname'), j.get('patronymic'), j.get('dateOfBirth'), sep=' ', end='.\n')
        # print(list[i].values())

def addNewPupil(list):
    initPupils(list)

def removePupil(list):
    dataPupil = input('Enter Surname, Firstname and Patronmic of pupil: ')
    dataPupil = dataPupil.split();
    delInd = []
    for i in range(0, len(list)):
        js = list[i].get('surname')
        jf = list[i].get('firstname')
        jp = list[i].get('patronymic')
        if js == dataPupil[0] and jf == dataPupil[1] and jp == dataPupil[2]:
            delInd.append(i)
    for i in range(0, len(delInd)):
        j = delInd[i]
        del list[j]

def sortPupils(list,key):
    for i in range(0, len(list)-1):
        if list[i].get('surname') > list[i+1].get('surname'):
            (list[i], list[i+1]) = (list[i+1], list[i])
    printValues(list)

def isBirth(list):
    today = datetime.datetime.today().strftime("%m-%d")
    check = 0
    for i in range(0, len(list)):
        s = list[i].get('dateOfBirth').rstrip()
        s = s[-5::1]
        if today == s:
            print("Today selebrate ", list[i].get('surname'), ' ', list[i].get('firstname'))
            check = 1
        if check == 0:
            print('There are no birthdays today')

def menu():
    print('Select the menu item: ')
    print('1 - review all values')
    print('2 - add new pupil')
    print('3 - remove pupil')
    print('4 - sort pupils')
    print('5 - birthDay')
    print('6 - exit')

    c = int(input('Enter your answer: '))
    return c

def chooseItem(list):

    choose = menu()
    if choose == 1:
        printValues(list)
        chooseItem(list)

    elif choose == 2:
        addNewPupil(list)
        chooseItem(list)

    elif choose == 3:
        removePupil(list)
        chooseItem(list)

    elif choose == 4:
        sortPupils(list, 'surname');
        chooseItem(list)

    elif choose == 5:
        isBirth(list);
        chooseItem(list)

    elif choose == 6:
        return
    else:
        print('Please, try again!');
        chooseItem(list)



def mainF():
    list = []
    choose = 0
    print("Enter list of pupils: ")
    for i in range(0,2):
        initPupils(list)
    chooseItem(list)

mainF()