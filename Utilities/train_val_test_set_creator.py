import os
import random as rand

#Dir where the test_set folder should be placed
#directory_1 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/'
directory_1 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/'

#Dir where the frames folder is located, for creation of multiple fractioned data sets multiple copies of frames folder is needed
#directory_2 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/test_set'
directory_2_1 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/all_labeled_frames_used_1/'
directory_2_2 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/all_labeled_frames_used_2/'
directory_2_3 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/all_labeled_frames_used_3/'
directory_2_4 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/all_labeled_frames_used_4/'

#one iteration per fractioned test set to be created
for j in range(4):
    directory_2 = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/all_labeled_frames_used_' + str(j+1) + '/'
    data_fraction = [0.25,0.5,0.75,1] #sizes of the data set fraction we want to create, from 25% of total to 100%
    os.chdir(directory_1)

    try:
        if not os.path.exists('data_set_'+str(data_fraction[j])): #add number for different test sets
            os.makedirs('data_set_'+str(data_fraction[j]))
            os.chdir(directory_1 + 'data_set_'+str(data_fraction[j]))
            os.makedirs('train_set_'+str(data_fraction[j]))
            os.makedirs('val_set_'+str(data_fraction[j]))
            os.makedirs('test_set_'+str(data_fraction[j]))
    except OSError:
        print('Error: Creating directory of data')

    # source directories for images
    dir1 = directory_2 + 'empty_set/'
    dir2 = directory_2 + 'person_set/'
    dir3 = directory_2 + 'dog_set/'
    dir4 = directory_2 + 'bike_set/'
    dir5 = directory_2 + 'empty_night_set/'
    dir6 = directory_2 + 'person_night_set/'

    dirs = [dir1, dir2, dir3, dir4, dir5, dir6]

    #destination directories where sets will be placed
    dest1 = directory_1 + 'data_set_'+str(data_fraction[j]) + '/train_set_'+str(data_fraction[j])+'/'
    dest2 = directory_1 + 'data_set_'+str(data_fraction[j]) + '/val_set_'+str(data_fraction[j])+'/'
    dest3 = directory_1 + 'data_set_'+str(data_fraction[j]) + '/test_set_'+str(data_fraction[j])+'/'


    for i in range(len(dirs)-2): #length of directory will include night images, len -2 will remove night images.
        curr_dir = dirs[i]
        os.chdir(curr_dir) #C:\Users\eitn35\Documents\EITN35_EVOLVE\image_frames\all_labeled_frames_used__1\empty_set

        count = 0
        wantedSplit = [0.64, 0.16, 0.2] #training, validation, test
        noExtract_train = len(os.listdir(curr_dir))*wantedSplit[0]*data_fraction[j] #train set
        noExtract_val = len(os.listdir(curr_dir))*wantedSplit[1]*data_fraction[j] #val set
        noExtract_test = len(os.listdir(curr_dir))*wantedSplit[2]*data_fraction[j] #test set

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

