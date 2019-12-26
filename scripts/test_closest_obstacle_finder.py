import cv2
import time
import os
import sys

sys.path.append("/Users/elijah/Dropbox/Programming/detectors/modules/")

import detectors
import input_output

video_path = ""
video_file = ""

photo_path = "/Users/elijah/Dropbox/Programming/detectors/images/"
photo_file = "two_objects.jpg"

def main ():
    img_source = input_output.Source (photo_path + photo_file)
    detector = detectors.Detector ('/Users/elijah/Dropbox/Programming/detectors/configs/closest_obstacle.json')
    
    while (True):
        frame_ = img_source.get_frame ()

        #frame = cv2.cvtColor (frame_, cv2.COLOR_RGB2BGR)
        frame = frame_
                
        cv2.waitKey (1)    

        (obstacle_pixels, labels), _ = detector.detect (frame, "obstacle detector")

        #draw obstacles on the frame
        result = frame.copy ()

        #os.system ("clear")
        #print (obstacle_pixels)

        #for i in range (len (obstacle_pixels)):
        #    x = i
        #    y = obstacle_pixels [i]

        #    type = labels [i]

        #    result = cv2.circle (result, (x, y), 5, (12 + type * 150, 250 - type * 120, 190 + type * 110), thickness = -1)

        stages = detector.get_stages_picts ("obstacle detector")
	
        for i in range (len (stages)):
            cv2.imshow (str (i), stages [i])
        
        #cv2.imshow ("frame", result)

        time.sleep (0.02)
       
        keyb = cv2.waitKey (1) & 0xFF
        
        if (keyb == ord('q')):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main ()