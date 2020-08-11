# --------------------------------------------------------
# This module is responsible for displaying menus
#
# By Kyle Mclean
# --------------------------------------------------------

import os

def mainMenu():
    # CLear the terminal
    os.system('clear')
    
    # Display introduction screen
    print(' This is a proof of concept for the reach bravo camera implimentation')
    print(' and has been developed for the TX2 board with ip camera. \n\n')
    
    # Welcome menu
    print('--------------------------------------------------------------------')
    print('\t Main menu')
    print('--------------------------------------------------------------------')
    print(' [1] Show raw camera feed')
    print(' [2] Change camera parameters')
    print(' [3] Calibrate camera')
    print(' [4] Exit program')
    val = input('\n Please input the number of your option: ')
    
    while val < 1 or val > 4:
        val = input('Invaild selection please try agian: ')
    
    os.system('clear')
    return val
    
def cameraMenu():
    os.system('clear')
    
    # Camera menu
    print('--------------------------------------------------------------------')
    print('\t Camera menu')
    print('--------------------------------------------------------------------')
    print(' [1] Change camera ip (e.g 192.168.1.88:554)')
    print(' [2] Return to main menu')
    val = input('\n Please input the number of your option: ')
    
    while val < 1 or val > 2:
        val = input('Invaild selection please try agian: ')
    
    os.system('clear')
    return val
    
def calibrationMenu():
    os.system('clear')
    
    # Calibration menu
    print('--------------------------------------------------------------------')
    print('\t Calibration menu')
    print('--------------------------------------------------------------------')
    print(' [1] Run calibration')
    print(' [2] Return to main menu')
    val = input('\n Please input the number of your option: ')
    
    while val < 1 or val > 3:
        val = input('Invaild selection please try agian: ')
    
    os.system('clear')
    return val
    
    
    
    
