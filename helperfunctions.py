import numpy as np
import cv2
from collections import deque


# # skin color detection

# In[2]:


# Function that detects whether a pixel belongs to the skin based on RGB values
# src - the source color image
# dst - the destination grayscale image where skin pixels are colored white and the rest are colored black
def mySkinDetect(src):
    # Surveys of skin color modeling and detection techniques:
    # 1. Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    # 2. Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    dst = np.zeros((src.shape[0], src.shape[1], 1), dtype = "uint8")
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            #b,g,r = src[i,j]
            b = int(src[i,j][0])
            g = int(src[i,j][1])
            r = int(src[i,j][2])
            if(r>95 and g>40 and b>20 and max(r,g,b)-min(r,g,b)>15 and abs(r-g)>15 and r>g and r>b):
                dst[i,j] = 255
    return dst


# # frame-to-frame differencing

# In[3]:


# Function that does frame differencing between the current frame and the first_frameious frame
# first_frame - the first_frameious color image
# curr - the current color image
# dst - the destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
# and first_frameious image are not the same
def myFrameDifferencing(hull, curr_frame):
    
    lastX = curr_frame[0]
    lastY = curr_frame[1]
    prev = (lastX, lastY)

    currX = hull[0][0][0]
    currY = hull[0][0][1]
    curr = (currX, currY)

    motion = (abs(currX - lastX) > 25 and lastX != 0) or (abs(currY - lastY) > 25 and lastY != 0)

    return motion, prev, curr

# # motion energy templates
# * example 1: the bottom row displays a cumulative binary motion energy image sequence corresponding to the frames above
# ![title](mh1.png)
# * example 2: pixel intensity is a function of the motion history at that location, where brighter values correspond to more recent motion, three actions: sit-down, arms-raise, crouch-down
# ![title](mh2.png)

# In[4]:


# Function that accumulates the frame differences for a certain number of pairs of frames
# mh - vector of frame difference images
# dst - the destination grayscale image to store the accumulation of the frame difference images
def myMotionEnergy(mh):
    # the window of time is 3
    mh0 = mh[0]
    mh1 = mh[1]
    mh2 = mh[2]
    dst = np.zeros((mh0.shape[0], mh0.shape[1], 1), dtype = "uint8")
    for i in range(mh0.shape[0]):
        for j in range(mh0.shape[1]):
            if mh0[i,j] == 255 or mh1[i,j] == 255 or mh2[i,j] == 255:
                dst[i,j] = 255
    return dst


# In[5]:


def main():
    # a) Reading a stream of images from a webcamera, and displaying the video
    # open the video camera no. 0
    # for more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
    cap = cv2.VideoCapture(0)
    
    #if not successful, exit program
    if not cap.isOpened():
        print("Cannot open the video cam")
        return -1

    # read a new frame from video
    success, first_frame_frame = cap.read()
    
    #if not successful, exit program
    if not success:
        print("Cannot read a frame from video stream")
        return -1
    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
    
    first_frame_frame = cv2.resize(first_frame_frame,(150,100))
    fMH1 = np.zeros((first_frame_frame.shape[0], first_frame_frame.shape[1], 1), dtype = "uint8")
    fMH2 = fMH1.copy()
    fMH3 = fMH1.copy()
    myMotionHistory = deque([fMH1, fMH2, fMH3]) 
    while(True):
        #read a new frame from video
        success, curr_frame = cap.read()
        curr_frame = cv2.resize(curr_frame,(150,100))
        if not success:
            print("Cannot read a frame from video stream")
            break
    
        cv2.imshow('frame',curr_frame)

        # b) Skin color detection
        mySkin = mySkinDetect(curr_frame)
        cv2.imshow('mySkinDetect',mySkin)

        # c) Background differencing
        frameDest = myFrameDifferencing(first_frame_frame, curr_frame)
        cv2.imshow('myFrameDifferencing',frameDest)

        # d) Visualizing motion history
        myMotionHistory.popleft()
        myMotionHistory.append(frameDest)
        myMH = myMotionEnergy(myMotionHistory)
        cv2.imshow('myMotionHistory',myMH)

        first_frame_frame = curr_frame
        
        # wait for 'q' key press. If 'q' key is pressed, break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return 0

def takeIMG():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
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

        cam.release()

        cv2.destroyAllWindows()

