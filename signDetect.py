import helperfunctions as hf
import numpy as np
import cv2
import time

# Open camera cap
cap = cv2.VideoCapture(1)

# Keep track os prev frame to frame differences
lastX = 0
currX = 0
lastY = 0
currY = 0
prev = (lastX, lastY)
curr = (currX, currY)

# Maintain number of pics outputted
img_counter = 0

while True:
    # Read frame from cap
    _, frame = cap.read()
    
    # Convert frame to HSV to match with skin range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_skin = np.array([0,20,70], dtype = np.uint8)
    upper_skin = np.array([20,255,255], dtype = np.uint8)

    # Preprocess the img (Match Skin, Erode, Dilate, Gaussian Blur)
    mask0 = cv2.inRange(hsv, lower_skin, upper_skin)   
    mask1 = cv2.erode(mask0, None, iterations = 2)
    mask2 = cv2.dilate(mask1, None, iterations = 4)
    mask3 = cv2.GaussianBlur(mask2, (5,5), 100)

    # Create Contours
    contours, hierarchy = cv2.findContours(mask3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    
    # Choose largest area contour
    ci = 0
    num_cnts = len(contours)
    if num_cnts != 0:
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>800):
                max_area=area
                ci=i
        cnt=contours[ci]

        # Draw convex hull
        hull = cv2.convexHull(cnt)

        # Maintain moments for motion
        moments = cv2.moments(cnt)
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00             
            centr=(cx,cy)       
        

        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull2 = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull2)

        i = 0
        mind=0
        maxd=0
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                dist = cv2.pointPolygonTest(cnt,centr,True)
                cv2.line(mask3,start,end,[255,0,0],2)                
                cv2.circle(mask3,far,10,[255,0,0],-1)
            print(i)

            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            
            waving, prev, curr = hf.myFrameDifferencing(hull, curr)
            if waving:
                cv2.putText(frame, "Waving", (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3, cv2.LINE_AA)
            
            if i >= 2 and i <= 3:
                cv2.putText(frame, "Scissors", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3, cv2.LINE_AA)
            elif i >= 4  and i <= 6:
                cv2.putText(frame, "Paper", (x+(w//3),y), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3, cv2.LINE_AA)
            elif i == 1:
                cv2.putText(frame, "Rock", (x+(w//3),y), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Please present hand sign", (0,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Please present hand sign", (0,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 3, cv2.LINE_AA)
    
    
    cv2.imshow("Detect Sign", frame)
    cv2.imshow("Mask3", mask3)

    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    
    
    if k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





