# --------------------------------------------------------
# This module manages the camera stream.
#
# The rtspCamera class contains all relevant methods and
# attributes to manage the camera feed and all corresponding
# image processing from the camera feed.
#
# By Kyle Mclean
# --------------------------------------------------------

# Import external modules
import numpy as np
import cv2
import multiprocessing as mp
import math

# Import interal modules
from VOManager import PinholeCamera2, VisualOdometry

# Define the rtsp camera class. This object can be used for all 
# image processing and connections

class rtspCamera():
    
    def __init__(self,rtsp_url,width,height):        
        
        # Define class attributes
        self.REQ_RESET = 0
        self.REQ_FRAME = 1
        self.REQ_CLOSE = 2
        
        # Load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))        
        
        # Start process
        self.p.daemon   = True
        self.p.start()
        
        # Define instance attributes
        self.name       = 'RSTP Camera Feed'
        self.width      = width
        self.height     = height
        self.edgeThresh = 100  
        
        # Load calibration files
        self.mtx = np.load('calibrationData/mtx.npy')
        self.dist = np.load('calibrationData/dist.npy')
        
    def end(self):      
        self.parent_conn.send(self.REQ_CLOSE)               # Send closure request to process

    def update(self,conn,rtsp_url):
        
        print("Loading camera connection ...")
        cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)     # Load into seperate process 
        
        run = True
        while run:
            cap.grab()                                      # Grab frames from the buffer
            rec_dat = conn.recv()                           # Recieve input data

            if rec_dat == self.REQ_FRAME:                   # If frame requested, read the next frame
                ret,frame = cap.read()
                conn.send(frame)

            elif rec_dat == self.REQ_CLOSE:                 # If colse requested, close the process
                cap.release()
                run = False
                     
        conn.close()

    def get_frame(self,resize=None):
        ### Used to grab frames from the cam connection process
        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase

        self.parent_conn.send(self.REQ_FRAME)               # Request frame
        frame = self.parent_conn.recv()                     # Get next frame from parent connection 

        
        self.parent_conn.send(self.REQ_RESET)               # Reset request 

        # Resize frame if needed
        if resize == None:            
            return frame
        else:
            return self.rescale_frame(frame,resize)

    def rescale_frame(self,frame, percent=65): 
        return cv2.resize(frame,None,fx=percent,fy=percent) # Rescale frame to desired value  
    
    def processCameraFeed(self):
        # Compute mapping
        h,  w = self.get_frame().shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),0,(w,h))
        mapx,mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,newcameramtx,(w,h),5)
        
        # Setup P3P algorithem
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        
        # Define offsets
        x0, y0 = 180, 20
         
        # Set window options       
        showWindow = 1
        showHelp = True
        unDist = False
        
        # Set text display options 
        font = cv2.FONT_HERSHEY_PLAIN
        helpText = "'Esc' to Quit, [1] Camera Feed, [2] Edge Detection, [3] Estimate camera pose, [4] Undistort image, [5] Hide help"
        
        while(True):
            if cv2.getWindowProperty(self.name, 0) < 0:             # Check to see if the user closed the window
                break;
                
            frame = self.get_frame()                                # Get the next frame: add "0.4" for reduced pixle count
            
            if unDist == True:
                dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
                x,y,w,h = roi
                frame = dst[y:y+h, x:x+w]
            
            if showWindow == 1:     # Show frame
                displayBuf = frame 
            
            elif showWindow == 2:   # Show edges
                hsv, blur, edges = self.egdeDetection(frame)
                displayBuf = edges
            
            elif showWindow == 3:   # Localise camera
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       # Apply colour filter (hsv)
                blur = cv2.GaussianBlur(gray,(7,7),1.5)             # Apply gussian blur
                
                ret, corners = cv2.findChessboardCorners(blur, (9,6), flags = cv2.CALIB_CB_FAST_CHECK)
                displayBuf = frame 
                
                if ret == True:
                    #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    # Find the rotation and translation vectors.
                    #rvecs, tvecs = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)
                    _, rvecs, tvecs = cv2.solvePnP(objp, corners, self.mtx, self.dist)
                    
                    X = tvecs[0]*26
                    Y = tvecs[1]*26
                    Z = tvecs[2]*26
                    
                    Theta = rvecs[0]*(180/np.pi)
                    Phi = rvecs[1]*(180/np.pi)
                    Phe = rvecs[2]*(180/np.pi)
                    
                    measurement = np.array([X,Y,Z,Theta,Phi,Phe], np.float32)
                    filtered = self.updateKalmanFilter(KF, measurement)
                    
                    count += 1
                    
                    X_proj, Y_proj = self.isoProjection(filtered[0],filtered[1],filtered[2])
                    draw_x_0, draw_y_0 = draw_x, draw_y 
                    draw_x, draw_y = (int(X_proj)//5+x0), (int(Y_proj)//5+y0)
                    if draw_x_0 != None:
                        #cv2.circle(traj, (draw_x,draw_y), 1, (count*255/4540,255-count*255/4540,0), 1)
                        cv2.line(traj, (draw_x_0, draw_y_0), (draw_x, draw_y), (255, 255, 0), thickness=2)
                    
                    pose = "X: %.1f [mm] Y: %.1f [mm] Z: %.1f [mm]  Theta: %.1f [deg] Phi: %.1f [deg] Phe: %.1f [deg] "%(X,Y,Z,Theta,Phi,Phe)
                    cv2.putText(displayBuf, pose, (11,40), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                    cv2.putText(displayBuf, pose, (10,40), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
                    
                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
                    displayBuf = self.draw(frame,corners,imgpts)
                    
                
                
                    
                x_offset = 0
                y_offset = 570
                displayBuf[y_offset:y_offset+traj.shape[0], x_offset:x_offset+traj.shape[1]] = traj
            
            if showHelp == True:
                cv2.putText(displayBuf, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(displayBuf, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            
            # Display final image screen
            cv2.imshow(self.name,displayBuf)
            
            key = cv2.waitKey(1)
                
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key == 49: # 1 key, show frame
                cv2.setWindowTitle(self.name,"Camera Feed")
                showWindow = 1
            elif key == 50: # 2 key, show Canny
                cv2.setWindowTitle(self.name,"Edge Detection")
                showWindow = 2
            elif key == 51: # 3 key, show Stages
                cv2.setWindowTitle(self.name,"Camera pose estimation")
                showWindow = 3
                
                # Intilise Kalman filter 
                KF = self.initKalmanFilter()
                
                # Setup trajectory tracking
                traj = np.zeros((150,250,3), dtype=np.uint8)
                count = 1      
                line_thickness = 2
                cv2.line(traj, (x0,y0), (100+x0, 0+y0), (255, 0, 0), thickness=line_thickness)
                cv2.line(traj, (x0,y0), (0+x0, 100+y0), (0, 255, 0), thickness=line_thickness)
                zX_proj, zY_proj = self.isoProjection(0,0,100)
                cv2.line(traj, (x0,y0), (int(zX_proj)+x0, int(zY_proj)+y0), (0, 0, 255), thickness=line_thickness)
                draw_x, draw_y = None, None     
                
            elif key == 52: # 4 key, toggle distroted
                unDist = not unDist 
            elif key == 53: # 6 key, toggle help
                showHelp = not showHelp
    
    def isoProjection(self,X,Y,Z):
        X_proj = X - Z*np.cos(-(180-135)*(np.pi/180))
        Y_proj = Y - Z*np.sin(-(180-135)*(np.pi/180))
        return X_proj, Y_proj
    
    def runVO(self):
        print('Starting visual odometry ...')
        frame = self.get_frame()
        cam = PinholeCamera2(frame.shape[1], frame.shape[0], self.mtx, self.dist)
        vo  = VisualOdometry(cam)
        
        # Init
        flag = False
        count = 1
        font = cv2.FONT_HERSHEY_PLAIN
        
        # Define offsets
        x0, y0 = 400, 50
        
        # Setup trajectory tracking
        traj = np.zeros((300,600,3), dtype=np.uint8)      
        line_thickness = 2
        cv2.line(traj, (x0,y0), (100+x0, 0+y0), (255, 0, 0), thickness=line_thickness)
        cv2.line(traj, (x0,y0), (0+x0, 100+y0), (0, 255, 0), thickness=line_thickness)
        zX_proj, zY_proj = self.isoProjection(0,0,100)
        cv2.line(traj, (x0,y0), (int(zX_proj)+x0, int(zY_proj)+y0), (0, 0, 255), thickness=line_thickness)
        draw_x, draw_y = None, None  
        
        while(True):
            if flag == True:
                count += 1
                frame = self.get_frame()
                displayBuf = frame
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       # Apply colour filter (hsv)
                img = cv2.GaussianBlur(gray,(7,7),1.5)              # Apply gussian blur
                   
	            
                if img is not None:
                	img_id = 0
                	vo.update(img, img_id)
                	
                cur_t = vo.cur_t
                cur_R = vo.cur_R
                anlges = np.array([0,0,0])

                if(count > 2):
                	z, y, x = cur_t[0], cur_t[1], cur_t[2]
                	# get angles from rotation matrix
                	angles = self.rotationMatrixToEulerAngles(cur_R)
                	n = angles[0]*(180/np.pi)
                	e = angles[1]*(180/np.pi)
                	d = anlges[2]*(180/np.pi)
                	kp = vo.detector.detect(img,None)
                	displayBuf = cv2.drawKeypoints(displayBuf, kp, displayBuf, color = (255,0,0))
                else:
                	x, y, z, n, e, d = 0., 0., 0., 0., 0., 0.

                #x, y = 0., 0.
                pose = "X: %.1f [??] Y: %.1f [??] Z: %.1f [??]  Theta: %.1f [deg] Phi: %.1f [deg] Phe: %.1f [deg] "%(x,y,z,n,e,d)
                cv2.putText(displayBuf, pose, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(displayBuf, pose, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
                
                X_proj, Y_proj = self.isoProjection(x,y,z)
                draw_x_0, draw_y_0 = draw_x, draw_y 
                draw_x, draw_y = (int(X_proj)+x0), (int(Y_proj)+y0)
                if draw_x_0 != None:
                    cv2.line(traj, (draw_x_0, draw_y_0), (draw_x, draw_y), (255, 255, 0), thickness=2)
                    
                x_offset = 0
                y_offset = img.shape[0] - traj.shape[0]
                displayBuf[y_offset:y_offset+traj.shape[0], x_offset:x_offset+traj.shape[1]] = traj
                
                
                cv2.imshow('Camera view', displayBuf)

                key = cv2.waitKey(1)
                
                if key == 27: # Check for ESC key
                    cv2.destroyAllWindows()
                    break;
                
                if cv2.getWindowProperty('Camera view', 0) < 0:             # Check to see if the user closed the window
                    break;

        
            flag = True 
        
        
    def isRotationMatrix(self, R) :
	    #checks if the output rotation matrix from feature tracking is a valid rotation matrix or not
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R) :
     	#Calculates rotation matrix to euler angles 

        assert(self.isRotationMatrix(R)) 
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
        
    def runCalibration(self):
        print('\nWelcome to the camera calibration module.To use this module you will need a 7x6 checker board pattern.\n')
        print('The camera feed has now started. Place your checker board in front of the camera and press [2] to take 10 shots at various different angles.')
        
        showHelp = True
        
        font = cv2.FONT_HERSHEY_PLAIN
        helpText = "'Esc' to Quit, [1] Camera feed, [2] Take next Shot, [4] Hide help"
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
        
        while(True):
            if cv2.getWindowProperty(self.name, 0) < 0: # Check to see if the user closed the window
                break;
                
            frame = self.get_frame() # Get the next frame: add "0.4" for reduced pixle count
            
            if showHelp == True:
                cv2.putText(frame, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(frame, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            
            
            key = cv2.waitKey(1)
                
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key == 49: # 1 key, show camera frame
                cv2.setWindowTitle(self.name,"Camera feed")
                
            elif key == 50: # 2 key, take next shot
                frame, ret, corners = self.calibrateCamera(frame) # Run take shot function
                
                if ret == True: 
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    print('Shots to go: ' + str(10 - len(objpoints)))
                
            elif key == 51: # 3 key, show Stages
                pass
                
            elif key == 52: # 4 key, toggle help
                showHelp = not showHelp
                        
            # Display final image screen
            cv2.imshow(self.name,frame)
            
            if len(objpoints) == 10:
                print('Calibrating camera ...')
                frame_size = (frame.shape[1], frame.shape[0])
                ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size,None,None)
                
                np.save('calibrationData/mtx.npy', self.mtx)
                np.save('calibrationData/dist.npy', self.dist)
                break
    
    def initKalmanFilter(self):
        
        nStates = 18
        nMeasurements = 6
        nInputs = 0
        
        dt = 1/20
        dt2 = dt*dt
        
        kalman = cv2.KalmanFilter(nStates, nMeasurements) # create KF
        
        # Define dynamic model
        kalman.transitionMatrix = np.array([[1, 0, 0, dt,  0,  0, dt2,   0,   0, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 1, 0,  0, dt,  0,   0, dt2,   0, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 1,  0,  0, dt,   0,   0, dt2, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  1,  0,  0,  dt,   0,   0, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  0,  1,  0,   0,  dt,   0, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  0,  0,  1,   0,   0,  dt, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   1,   0,   0, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   1,   0, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   1, 0, 0, 0,  0,  0,  0,   0,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 1, 0, 0, dt,  0,  0, dt2,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 1, 0,  0, dt,  0,   0, dt2,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 1,  0,  0, dt,   0,   0, dt2],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 0,  1,  0,  0,  dt,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 0,  0,  1,  0,   0,  dt,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 0,  0,  0,  1,   0,   0,  dt],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 0,  0,  0,  0,   1,   0,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 0,  0,  0,  0,   0,   1,   0],
                                            [0, 0, 0,  0,  0,  0,   0,   0,   0, 0, 0, 0,  0,  0,  0,   0,   0,   1]], np.float32)

        # Define measurement model
        kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], np.float32)

        kalman.processNoiseCov = np.identity(nStates,np.float32)*1e-2
        kalman.measurementNoiseCov = np.identity(nMeasurements,np.float32)*1e-1

        return kalman

    
    def updateKalmanFilter(self, KF, measurement):

        # First predict, to update the internal statePre variable
        prediction = KF.predict()
        
        # The "correct" phase that is going to use the predicted value and our measurement
        return KF.correct(measurement)
        
              
             
    def calibrateCamera(self, frame):    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Make frame grayscale
        ret, corners = cv2.findChessboardCorners(gray, (7,7),None) # Find the chess board corners

        # If corners are found
        if ret == True:    
            corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) # Refine the corners using sup pixles
            frame = cv2.drawChessboardCorners(frame, (7,7), corners,ret) # Draw corners on frame
            
        return frame, ret, corners

    def openWindow(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.moveWindow(self.name, 0, 0)
        cv2.setWindowTitle(self.name, 'Bravo Camera Feed')

    def destroyOpenCV(self):  
        cv2.destroyAllWindows()        
        
    def egdeDetection(self,frame):
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Apply gray scale filter
        blur  = cv2.GaussianBlur(hsv,(7,7),1.5)         # Apply gussian blur (better edge detection)
        edges = cv2.Canny(blur,0,self.edgeThresh)       # Apply edge detection on blured image 
        
        return hsv, blur, edges   
        
    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img
    

            
            
