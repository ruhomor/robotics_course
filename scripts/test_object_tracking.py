import detectors
import cv2
import time
import os
import math

#TODO: implement class, incapsulating input source
#possible inputs: video, camera, photo, ROS (Volodya sexflexx)

CAMERA = 0
VIDEO  = 1
PHOTO  = 2 

video_path = ""
video_file = ""

photo_path    = "/Users/elijah/Dropbox/Programming/detectors/images/"
photo_file    = "basket/2.jpg"
obstacle_file = "obstacle_only.jpg"

output_path = "/Users/elijah/Dropbox/Programming/RoboCup/nao_cv/geometrical/chessboard_images/"

def main ():
    INPUT_SOURCE = PHOTO

    #cam_num = max (get_available_cameras ())

    #cam = cv2.VideoCapture (cam_num)

    #if (INPUT_SOURCE != CAMERA):
    #    cam.release ()

    #if (INPUT_SOURCE == VIDEO):
    #    cam = cv2.VideoCapture (video_path + video_file)

    if (INPUT_SOURCE == PHOTO):
        img = cv2.imread (photo_path + photo_file)

    obstacle = cv2.imread (photo_path + obstacle_file)
    obstacle_sh = obstacle.shape

    str_num = 0
    
    detector = detectors.Detector ('/Users/elijah/Dropbox/Programming/detectors/configs/object_tracking.json')
    
    sh = img.shape
    x_obs_rot = sh [1] // 2
    y_obs_rot = sh [0] // 2
    radius = x_obs_rot // 2
    angle = 0

    while (True):
        #if (INPUT_SOURCE == CAMERA or INPUT_SOURCE == VIDEO):
        #    ret, frame_ = cam.read ()

        if (INPUT_SOURCE == PHOTO):
            frame_ = img.copy ()

        x_obs = x_obs_rot + radius * math.cos (angle)
        y_obs = y_obs_rot + radius * math.sin (angle)

        angle += 0.1

        frame [x_obs : x_obs + obstacle_sh [1],
               y_obs : y_obs + obstacle_sh [0], :] = obstacle

        frame = frame_
                
        cv2.waitKey (1)    
        os.system ('clear')
        
        (x, y), success = detector.detect (frame, "obstacle detector")

        #draw circle on the frame
        if (success == True):
            print ("detected")
            
            result = cv2.circle (result, (x, y), 9, (120, 15, 190), thickness = -1)
            
        else:
            print ("not detected")

        stages = detector.get_stages ()
	
        for i in range (2):
            cv2.imshow (str (i), stages [i])

        #processing_stages = detector.stages ()
	
	#resultant_frame = form_images (processing_stages)
        
        cv2.imshow ("frame", result)

        time.sleep (0.02)

        #clear_output (wait=True)
        
        keyb = cv2.waitKey (1) & 0xFF
        
        if (keyb == ord('q')):
            break

    cam.release ()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main ()