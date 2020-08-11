import numpy as np 
from glob import glob
import os
import os.path
import cv2
import math
import csv

from visual_odometry import PinholeCamera, VisualOdometry

FILE_PATH = '/home/bravo/Documents/blueprintCamVis/monoVisInertOd/VO/dataSet/mav0/cam1/data/'
    



def rename_images():
#import os
#import os.path
#renames images for a dataset for easy import while maintaining the order
	img_types = ['.png']
	for dirpath, dirnames, fnames in os.walk(path):
		imgs = [f for f in fnames if os.path.splitext(f)[1] in img_types]
		imgs.sort()
		for j, im in enumerate(imgs):
			name, ext = os.path.splitext(im)
			os.rename(os.path.join(dirpath, im), os.path.join(dirpath,'{}.{}'.format(j, ext)))

def read_image():
	images = []
	img_mask = "/home/karan/datasets/mav0/cam1/data/*.png"
	for fn in glob(img_mask):
		img = cv2.imread(fn, -1)
		if img is not None:
			images.append(img)
	return images

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
    

if __name__ == '__main__':
    
    flag = False
    count = 1
   
    cam = PinholeCamera(752.0, 480.0, 458.654, 457.296, 367.215, 248.375)
    vo = VisualOdometry(cam)

    traj = np.zeros((600,600,3), dtype=np.uint8)
    
    myfilepath = '/home/bravo/Documents/blueprintCamVis/monoVisInertOd/VO/dataSet/mav0/cam0/data.csv'
    with open(myfilepath, 'rb') as f:
        mycsv = csv.reader(f)
        for row in mycsv:
            img_id = row[1]

            if flag == True:
                count += 1
                img = cv2.imread(FILE_PATH + img_id, 0)
               
	        
            	if img is not None:
            		vo.update(img, img_id)
            	
            	cur_t = vo.cur_t
            	cur_R = vo.cur_R
            	anlges = np.array([0,0,0])
            	n=0.
            	e=0.
            	d=0.
            	
            	if(count > 2):
            		x, y, z = cur_t[0], cur_t[1], cur_t[2]
            		#get angles from rotation matrix
            		angles = rotationMatrixToEulerAngles(cur_R)
            		n = angles[0]
            		e = angles[1]
            		d = anlges[2]
            	else:
            		x, y, z = 0., 0., 0.
            		
            	draw_x, draw_y = int(z)+300, int(0)+300
            	#true_x, true_y = int(vo.trueX)+300, int(vo.trueZ)+300

            	cv2.circle(traj, (draw_x,draw_y), 1, (count*255/4540,255-count*255/4540,0), 1)
            	#cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
            	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
            	text = "Coordinates: x=%.1fm y=%.1fm z=%.1fm '\n' n=%.1f e=%.1f d=%.1f "%(x,y,z,n,e,d)
            	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

            	cv2.imshow('Camera view', img)
            	cv2.imshow('Trajectory', traj)
                cv2.waitKey(1)
    
            flag = True 

    cv2.imwrite('map.png', traj)












