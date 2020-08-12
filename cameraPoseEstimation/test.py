import numpy as np 
from glob import glob
import os
import os.path
import cv2
import math
import csv

from manager.VOManager import PinholeCamera, VisualOdometry

IMG_FILE_PATH = '/home/bravo/Documents/blueprintCamVis/cameraPoseEstimation/dataSet/mav0/cam0/data/'
FNM_FILE_PATH = '/home/bravo/Documents/blueprintCamVis/cameraPoseEstimation/dataSet/mav0/cam0/data.csv'


def isRotationMatrix(R) :
	#checks if the output rotation matrix from feature tracking is a valid rotation matrix or not
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
 	#Calculates rotation matrix to euler angles 

    assert(isRotationMatrix(R)) 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
    
def isoProjection(X,Y,Z):
    X_proj = X - Z*np.cos(-(180-135)*(np.pi/180))
    Y_proj = Y - Z*np.sin(-(180-135)*(np.pi/180))
    return X_proj, Y_proj
    

if __name__ == '__main__':
    
    flag = False
    count = 1
   
    #cam = PinholeCamera(752.0, 480.0, 458.654, 457.296, 367.215, 248.375)
    cam = PinholeCamera(752.0, 480.0, 458.654, 457.296, 367.215, 248.375,-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)
    vo  = VisualOdometry(cam)
    
    # Define offsets
    x0, y0 = 300, 300
    
    # Setup trajectory tracking
    traj = np.zeros((600,600,3), dtype=np.uint8)      
    line_thickness = 2
    cv2.line(traj, (x0,y0), (100+x0, 0+y0), (255, 0, 0), thickness=line_thickness)
    cv2.line(traj, (x0,y0), (0+x0, 100+y0), (0, 255, 0), thickness=line_thickness)
    zX_proj, zY_proj = isoProjection(0,0,100)
    cv2.line(traj, (x0,y0), (int(zX_proj)+x0, int(zY_proj)+y0), (0, 0, 255), thickness=line_thickness)
    draw_x, draw_y = None, None  
    
    with open(FNM_FILE_PATH, 'rb') as f:
        mycsv = csv.reader(f)
        for row in mycsv:
            img_id = row[1]

            if flag == True:
                count += 1
                img = cv2.imread(IMG_FILE_PATH + img_id, 0)
               
	        
            	if img is not None:
            		vo.update(img, img_id)
            	
            	cur_t = vo.cur_t
            	cur_R = vo.cur_R
            	anlges = np.array([0,0,0])

            	if(count > 2):
            		x, y, z = cur_t[0], cur_t[1], cur_t[2]
            		# get angles from rotation matrix
            		angles = rotationMatrixToEulerAngles(cur_R)
            		n = angles[0]
            		e = angles[1]
            		d = anlges[2]
            		kp = vo.detector.detect(img,None)
            		img = cv2.drawKeypoints(img, kp, img, color = (255,0,0))
            	else:
            		x, y, z, n, e, d = 0., 0., 0., 0., 0., 0.

            	#x = 0
            	X_proj, Y_proj = isoProjection(x,y,z)
                draw_x_0, draw_y_0 = draw_x, draw_y 
                draw_x, draw_y = (int(X_proj)+x0), (int(Y_proj)+y0)
                if draw_x_0 != None:
                    cv2.line(traj, (draw_x_0, draw_y_0), (draw_x, draw_y), (255, 255, 0), thickness=2)

#            	text = "Coordinates: x=%.1fm y=%.1fm z=%.1fm '\n' n=%.1f e=%.1f d=%.1f "%(x,y,z,n,e,d)
#            	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

            	cv2.imshow('Camera view', img)
            	cv2.imshow('Trajectory', traj)
                cv2.waitKey(1)
                
                if cv2.getWindowProperty('Camera view', 0) < 0:             # Check to see if the user closed the window
                    break;
                
                if cv2.getWindowProperty('Trajectory', 0) < 0:             # Check to see if the user closed the window
                    break;
    
            flag = True 
            



                    
                    







