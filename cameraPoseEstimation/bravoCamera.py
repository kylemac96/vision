#!/usr/bin/env python

# --------------------------------------------------------
# Main file for the camera pose estimation package.
#
# This module instansates a new camera instance and runs
# the main program implimentation. 
#
# By Kyle Mclean
# --------------------------------------------------------

# Import external modules
import sys
import time


# Import local modules
import system.systemMenus
import manager.cameraManager

# Define global parameters for this module
IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080

# Menu states
MAIN_MENU    = 0
OPEN_CV      = 1
CALIBRATE    = 2
CAMERA_MENU  = 3
CAL_MENU     = 4

# Bye string
s1 = '  ____               ______   _ _      _       _ \n'
s2 = ' |  _ \             |  ____| | (_)    (_)     | |\n'
s3 = ' | |_) |_   _  ___  | |__ ___| |_  ___ _  __ _| |\n'
s4 = ' |  _ <| | | |/ _ \ |  __/ _ \ | |/ __| |/ _` | |\n'
s5 = ' | |_) | |_| |  __/ | | |  __/ | | (__| | (_| |_|\n'
s6 = ' |____/ \__, |\___| |_|  \___|_|_|\___|_|\__,_(_)\n'
s7 = '         __/ |                                   \n'
s8 = '        |___/                                    \n'

# Default uri for rtsp camera
# DEFAULT_RTSP_URI = 'rtsp://192.168.1.88:554'
DEFAULT_RTSP_URI = 'rtsp://192.168.1.204:554'

def main():
    rtsp_uri = DEFAULT_RTSP_URI
    
    # Initilisation of program state
    state = MAIN_MENU
    
    # While the program has not been terminated by user
    while True:
        if state == MAIN_MENU:
            # Display main menu
            idx = system.systemMenus.mainMenu()
                
            if idx == 1:
                state = OPEN_CV
            elif idx == 2:
                state = CAMERA_MENU               
            elif idx == 3:
                state = CAL_MENU
            else:
                # Terminate program
                sys.exit('Program terminated \n' + s1 + s2+ s3+ s4+ s5+ s6 +s7 +s8)
        
        elif state == CAMERA_MENU:
            # Display camera menu
            idx = system.systemMenus.cameraMenu()
            
            if idx == 1:
                # CHANGE IP ADDRESS
                print('\nInput new camera ip (e.g 192.168.1.88:554) below')
                ip = raw_input('\nInput IP address: ')
                rtsp_uri = 'rtsp://' + ip
                
            elif idx == 2:
                state = MAIN_MENU
        
        elif state == CAL_MENU:
            # Display calibration menu
            idx = system.systemMenus.calibrationMenu()
            
            if idx == 1:
                state = CALIBRATE  
            elif idx == 2:
                state = MAIN_MENU
            elif idx == 3:
                state = MAIN_MENU

        elif state == OPEN_CV:
            # Create camera object and connect to camera
            cam = manager.cameraManager.rtspCamera(rtsp_uri, IMAGE_WIDTH, IMAGE_HEIGHT)
            
            # Hand over to openCV
            cam.openWindow()
            cam.processCameraFeed()
            cam.destroyOpenCV()
            cam.end() # Release camera object?           

            # Hand back to main menu
            state = MAIN_MENU           
            
        elif state == CALIBRATE:
            # Create camera object and connect to camera
            cam = manager.cameraManager.rtspCamera(rtsp_uri, IMAGE_WIDTH, IMAGE_HEIGHT)
            
            # Hand over to openCV
            cam.openWindow()
            cam.runCalibration()
            cam.destroyOpenCV()
            cam.end() # Release camera object?         

            # Hand back to main menu
            state = MAIN_MENU     
            
    
if __name__ == '__main__':
    main()
    

    
    

    
    
