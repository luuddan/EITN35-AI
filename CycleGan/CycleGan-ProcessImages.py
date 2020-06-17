from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed


# load all images in a directory into memory
def load_images(path, size=(208, 208)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
    return asarray(data_list)


# dataset path
path = 'C:/Users/eitn35/Documents/EITN35/video_files/'
# load dataset A
dataA1 = load_images(path + 'frames/SprayDayXSmall/')
print('Loaded dataA: ', dataA1.shape)
# load dataset B
dataB1 = load_images(path + 'frames_sprayed_XSmall/')
print('Loaded dataB: ', dataB1.shape)
# save as compressed numpy array
filename = 'day2spray208XSmall.npz'
savez_compressed(filename, dataA1, dataB1)
print('Saved dataset: ', filename)

# load and plot the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('day2spray208XSmall.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(dataB[i].astype('uint8'))
pyplot.show()