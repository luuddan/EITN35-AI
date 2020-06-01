import os
import random as rand

#Dir where the test_set folder should be placed
#directory_1 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/'
directory_1 = 'C:/Users/eitn35/Documents/EITN35 EVOLVE/image_frames/'

#Dir where the frames folder is located
#directory_2 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/test_set'
directory_2 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/test_set'
os.chdir(directory_1)

try:
    if not os.path.exists('test_set'):
        os.makedirs('test_set')
        os.chdir(directory_2)
        os.makedirs('person_set')
        os.makedirs('bike_set')
        os.makedirs('dog_set')
        os.makedirs('empty_set')
except OSError:
    print('Error: Creating directory of data')

os.chdir(directory_1 + 'train_set')

count = 0
wantedSplit = [0.8, 0.1, 0.1] #training, validation, test
noExtract = len(os.listdir(directory_2))*wantedSplit[2]
PRINT_DEBUG = True

persons = 0
bikes = 0
dogs = 0
emptys = 0
done = False

while not done:
    index = rand.randint(0, len(os.listdir(directory_1 + 'train_set')))
    img_pick = os.listdir(directory_1 + 'train_set').sort()[index]
    no_persons =

while count < noExtract:
    index = rand.randint(1, len(os.listdir(directory_2)))
    file_list = os.listdir(directory_2)

    #if PRINT_DEBUG : print(str(file_list[index]) + " exported to test_set...")

    os.rename(
        directory_2 + str(file_list[index]),
        directory_1 + '/test_set/' + str(file_list[index])
    )
    count += 1

print(str(noExtract) + " files exported to test_set")

