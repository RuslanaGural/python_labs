from module_1_gural import game

int(input('Enter namber from 0 to 10: '))

game()

switchi = 1

while (switchi == 1):
    userQuation = input('You want to try again? ')
    if (userQuation == 'y' or userQuation == 'Y'):
        game()
    else:
        switchi = 0
        print('Game over!')
