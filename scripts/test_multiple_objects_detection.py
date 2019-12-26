
import cv2
import time
import os
import sys

sys.path.append("../modules")

import detectors
#TODO: implement class, incapsulating input source
#possible inputs: video, camera, photo

CAMERA = 0
VIDEO  = 1
PHOTO  = 2

video_path = ""
video_file = ""

photo_path = "/home/i/detectors/images/2019_08_11_08h00m33s"
#photo_file = "basket/2.jpg"
photo_file = "00014.png"

#output_path = "/Users/elijah/Dropbox/Programming/RoboCup/nao_cv/geometrical/chessboard_images/"
output_path = ''
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

    str_num = 0
    
    detector = detectors.Detector ('multiple_objects.json')
    
    while (True):
        #if (INPUT_SOURCE == CAMERA or INPUT_SOURCE == VIDEO):
        #    ret, frame_ = cam.read ()

        if (INPUT_SOURCE == PHOTO):
            frame_ = img.copy ()

        #frame = cv2.cvtColor (frame_, cv2.COLOR_RGB2BGR)
        frame = frame_
                
        cv2.waitKey (1)    
        os.system ('clear')
        
        (bbox_tl, bbox_br), success = detector.detect (frame, "ball detector")
        result = frame.copy ()

        if (success == True):
            print ("detected")
            
            result = cv2.rectangle (frame.copy (), bbox_tl, bbox_br, (255, 0, 0), 5)

        else:
            print ("not detected")

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
