import cv2
import numpy as np
import os

# Dir where video files to be split are located
#directory_1 = '../../Documents/EITN35/video_files/converted'
directory_1 = 'C:/Users/eitn35/Documents/EITN35/video_files/Archive'
# Dir where frames should be saved
#directory_2 = '../../Documents/EITN35/video_files'

directory_2 = 'C:/Users/eitn35/Documents/EITN35/video_files'
os.chdir(directory_2)

# Display time stamps of saved frames and progress
PRINT_DEBUG = True

# Create "frames" folder if it does not exist
try:
    if not os.path.exists('frames'):
        os.makedirs('frames')
except OSError:
    print('Error: Creating directory of data')


# Splitting one video file. Input: video fiel, Output: frames
def split_video(video_file):
    vidcap = cv2.VideoCapture(directory_1 + video_file)
    sec = 0
    frameRate = 0.25  # number of seconds between each capture
    count = 1
    vidcap.get(cv2.CAP_PROP_POS_MSEC)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            date_stamp = video_file.split('_')[2]
            time_stamp = video_file.split('_')[3].split('.')[0] + "+" + str(sec)
            if PRINT_DEBUG: print("Time stamp " + time_stamp)

            # save frame as JPG file
            cv2.imwrite("./frames/frame_"+date_stamp+"_"+time_stamp+"nr"+str(count)+"_"+".jpg", image)
            #cv2.imwrite("./frames/frame_" + str(count) + ".jpg", image)
        return hasFrames

    success = getFrame(sec)
    while success and (sec*1000< vidcap.get(cv2.CAP_PROP_POS_MSEC)):
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

    # When everything done, release the capture
    vidcap.release()
    cv2.destroyAllWindows()

#Dir where video files to be split are located


for video_file in os.listdir(directory_1):
    #if(video_file.endswith("mp4")):
    if PRINT_DEBUG: print("Starting splitting of " + video_file + "...")
    split_video('/' + video_file)
    if PRINT_DEBUG: print("Splitting of " + video_file + " done.")