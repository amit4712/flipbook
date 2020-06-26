import cv2
import numpy as np
import imutils

background=None

# Background subtraction via running average method
def running_avg(image):
    global background

    if background is None:
        background=np.float32(image.copy())
        return

    alpha=0.5
    cv2.accumulateWeighted(image, background, alpha)


#To segment the region of hand in the image
def segment(image, threshold=25):
    global background

    difference=cv2.absdiff(np.uint8(background), image)
    thresholded=cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #return None, if no contours detected
    if len(cnts)==0:
        return
    else:
        #Get the maximum contour i.e hand area
        segmented = max(cnts, key=cv2.contourArea)
        return(thresholded, segmented)


#Main Function
capture=cv2.VideoCapture(0)
# region of interest (ROI) coordinates
top, right, bottom, left = 10, 350, 225, 590
num_frames=0
while True:
    _, frame=capture.read()
    frame=imutils.resize(frame, width=700)
    frame=cv2.flip(frame, 1)
    clone=frame.copy()
    (h,w)=frame.shape[:2]
    roi = frame[top:bottom, right:left]
    gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (7,7), 0)

    if num_frames<30:
        running_avg(gray)
    else:
        hand=segment(gray)

        if hand is not None:
            (thresholded, segmented)= hand
            #drawing segmented region and displaying it in a frame
            cv2.drawContours(clone, [segmented+(right, top)] , -1, (0,0,255))
            cv2.imshow("Thresholded", thresholded)

    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    num_frames +=1

    #displaying frame with segmented hand
    cv2.imshow("Video Feed", clone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
