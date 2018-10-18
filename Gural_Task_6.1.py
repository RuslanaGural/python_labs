#Створення прикладних програм на мові Python. Лабораторна робота №6.1. Гураль Руслана. FI-9119


print('''' Створення прикладних програм на мові Python. Лабораторна робота №6.1
  Гураль Руслана. FI-9119''')

enterList = input('Enter list: ')
length = len(enterList)
myList = []

def numberString(stringList, resultList):
    length = len(stringList)
    i = 0
    while i < length:
        s_int = ''
        element = stringList[i]
        while '0' <= element <= '9':
            s_int += element
            i += 1
            if i < length:
                element = stringList[i]
            else:
                break
        i += 1
        if s_int != '':
            resultList.append(int(s_int))

    return resultList

numberString(enterList,myList)

for i in range(0,len(myList)-1,2):
    (myList[i], myList[i+1]) = (myList[i+1], myList[i])


print(myList)