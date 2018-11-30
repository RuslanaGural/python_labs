from random import randint

def game ():
    numb = randint(1,10)

    try:
        userNumb = int(input('Enter namber from 0 to 10: '))
    except:
        print('Please enter an integer number:  ')
        userNumb = int(input('Enter namber from 0 to 10: '))

    if (numb == userNumb):
        print("You win!")
    else:
        print('Upsss! You lose')