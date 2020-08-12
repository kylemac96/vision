# Camera pose estimation:

This package is an interface for the nvidia JETSON TX2 with a IMX385 ip camera 
to achieve camera pose estimation. The system impliments the following features:

 - Camera connection
 - Camera calibration
 - Pose estimation using the perspective-n-points method

Working on:
 - Visual odometry (no IMU data at this stage)


# TODO:

Some things that still need to be done:

 - Turn the 3d plot into a function that is a part of .self
 - *** Visual odometry *** in test 
 - Apply the funtion for 3d plots and compare with bench marks
 - Add test into the main script as a file
 - Add explination about the visual odometry and why only a bench mark file can be used
 
 - Apply Kalman filter to visual odomety 
 - Use IMU data
 - Work on attaching IMU to camera (could use IMU-6050)
 - Need to calibrte IMU if this method is used (research IMU calibration techniques/invesatigate the method implimented in matlab for a python port?????)
 
 
 - Electrical schematics for mounting an IMU
 - Investigate global shutter cameras for this implimentation
 
 - Object recognition 
 - How can we cast points on to a 3d surface and estimate depth from a know camera translation? Is this even possible?
 - 

# Random links:
https://jkjung-avt.github.io/tx2-camera-with-python/
https://jkjung-avt.github.io/opencv3-on-tx2/
https://github.com/luigifreda/pyslam
