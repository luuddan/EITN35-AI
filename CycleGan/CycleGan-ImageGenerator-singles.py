
# example of using saved cyclegan models for image translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

def select_all(dataset, n_samples):
	# retrieve selected images
	X = []
	for i in range(n_samples):
		X.append(dataset[[i]])
	return X


# plot the image, the translation, and the reconstruction
def show_plot(imagesX):
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
A_data, B_data = load_real_samples('../../../Documents/EITN35/night2day208testSetNightFinal.npz')
print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('../../../Desktop/CycleGanBigFrameSetNoBikesV1/g_model_AtoB_019740_D2N208BigFrameSetNoBikesV1.h5', cust)
model_BtoA = load_model('../../../Desktop/CycleGanBigFrameSetNoBikesV1/g_model_AtoB_019740_D2N208BigFrameSetNoBikesV1.h5', cust)


A_real = select_all(A_data, 272)
print(A_real[1].shape)
i = 0
counter = 0
for e in A_real:
	#e= (e + 1)/2.0
	A_generated  = model_BtoA.predict(e)
	A_generated = (A_generated + 1)/2.0
	fig = pyplot.figure(frameon=False)
	ax = pyplot.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	pyplot.axis('off')
	fig.add_axes(ax)
	ax.imshow(A_generated[0], aspect='auto')

	filename1 = '../../../Documents/EITN35/video_files/frames/CycleGan//TestSetCycleNight/GeneratedDayFinal/%s_generated_pic_D2N208BigFrameSetV2.png' % (counter)
	fig.savefig(filename1)
	counter += 1

pyplot.close()
