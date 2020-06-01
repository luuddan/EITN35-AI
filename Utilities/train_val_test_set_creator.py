import os
import random as rand

#Dir where the test_set folder should be placed
#directory_1 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/'
directory_1 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/'

#Dir where the frames folder are located
#directory_2 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/test_set'
directory_2 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/all_labeled_frames_used/'

os.chdir(directory_1)

try:
    if not os.path.exists('test_set'):
        os.makedirs('train_set')
        os.makedirs('val_set')
        os.makedirs('test_set')
except OSError:
    print('Error: Creating directory of data')

dir1 = directory_2 + 'person_set/'
dir2 = directory_2 + 'person_night_set/'
dir3 = directory_2 + 'dog_set/'
dir4 = directory_2 + 'bike_set/'
dir5 = directory_2 + 'empty_set/'
dir6 = directory_2 + 'empty_night_set/'

dirs = [dir1, dir2, dir3, dir4, dir5, dir6]

dest1 = directory_1 + 'train_set/'
dest2 = directory_1 + 'val_set/'
dest3 = directory_1 + 'test_set/'

for i in range(len(dirs)):
    curr_dir = dirs[i]
    os.chdir(curr_dir)

    count = 0
    wantedSplit = [0.64, 0.16, 0.2] #training, validation, test
    noExtract_train = len(os.listdir(curr_dir))*wantedSplit[0] #train set
    noExtract_val = len(os.listdir(curr_dir))*wantedSplit[1] #val set
    noExtract_test = len(os.listdir(curr_dir))*wantedSplit[2] #test set

    PRINT_DEBUG = True

    while count < noExtract_train-1:
        index = rand.randint(1, len(os.listdir(curr_dir))-1)
        file_list = os.listdir(curr_dir)

        os.rename(
            curr_dir + str(file_list[index]),
            dest1 + str(file_list[index])
        )
        count += 1

    count = 0

    while count < noExtract_val-1:
        index = rand.randint(1, len(os.listdir(curr_dir))-1)
        file_list = os.listdir(curr_dir)

        os.rename(
            curr_dir + str(file_list[index]),
            dest2 + str(file_list[index])
        )
        count += 1

    count = 0

    while count < noExtract_test-1:
        index = rand.randint(1, len(os.listdir(curr_dir))-1)
        file_list = os.listdir(curr_dir)

        os.rename(
            curr_dir + str(file_list[index]),
            dest3 + str(file_list[index])
        )
        count += 1

#print(str(noExtract) + " files exported to test_set")

