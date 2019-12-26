import detectors
import cv2
import time
import os

#TODO: implement class, incapsulating input source
#possible inputs: video, camera, photo

CAMERA = 0
VIDEO  = 1
PHOTO  = 2

video_path = ""
video_file = ""

photo_path = "/Users/elijah/Dropbox/Programming/kondo/vision/images/"
photo_file = "basket/2.jpg"
#photo_file = "no_basket.png"

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

    #cv2.namedWindow ("frame", cv2.WINDOW_NORMAL)
    #cv2.namedWindow ("frame")
    
    #cv2.resizeWindow ("frame", (640*2, 480*2))
    #cv2.resizeWindow ("frame", (480, 640))

    str_num = 0

    #low_th  = (57, 150, 110)
    #high_th = (67, 160, 120)

    #detector = detectors.Detector ()
    #detector.add_filter (detectors.inrange (low_th, high_th), "inrange")
    #detector.add_filter (detectors.max_area_cc_bbox (), "bbox extraction")
    
    detector = detectors.Detector ('basket_detector.txt')
    
    while (True):
        #if (INPUT_SOURCE == CAMERA or INPUT_SOURCE == VIDEO):
        #    ret, frame_ = cam.read ()

        if (INPUT_SOURCE == PHOTO):
            frame_ = img.copy ()

        #frame = cv2.cvtColor (frame_, cv2.COLOR_RGB2BGR)
        frame = frame_
                
        cv2.waitKey (1)    
	
	#top left, bottom right
        (bbox_tl, bbox_br), success = detector.detect (frame)
	#draw bbox on the frame
        result = cv2.rectangle (frame.copy (), bbox_tl, bbox_br, (255, 0, 0), 5)

        #bottom point coordinates
        #(x, y), success = detector.detect (frame)

        result = frame.copy ()

        os.system ('clear')

        #draw circle on the frame
        if (success == True):
            print ("detected")
            
            #result = cv2.circle (result, (x, y), 5, (120, 150, 190), thickness = -1)
            result = cv2.rectangle (frame.copy (), bbox_tl, bbox_br, (255, 0, 0), 5)

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