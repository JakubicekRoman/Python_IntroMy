import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

# import cv2

# a = np.array([5, 6,3, 5])
# b = np.array([0,5,1,6])
# vel = a.shape
# %timeit np.sum(a)

# img = cv2.imread('img1.jpg')
# cv2.imshow('a frame', img)	# a frame is the title name of the display window
# cv2.waitKey(0)
# cv2.destroyAllWindows()


from skimage.color import rgb2gray

imgTrain = im.imread('img_train.jpg','jpg')
# plt.figure()
# plt.imshow(img,cmap='gray')
# plt.show()

# grey = rgb2gray(img)
# grey = img[:,:,2]/255
greyTrain = imgTrain/255
# greyTrain = greyTrain[:,:,2]

plt.figure()
plt.imshow(greyTrain,cmap='gray')
plt.show()

histogram, bin_edges = np.histogram(grey, bins=256, range=(0, 1))
plt.figure()
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()

BWTrain = greyTrain < 0.7
BWTrain = BWTrain[:,:,2]

# plt.figure()
# plt.imshow(BWTrain,cmap='gray')
# plt.imshow(BWTrain,cmap='gray')
# plt.show()


imgTest = im.imread('img_test.jpg','jpg')
plt.figure()
plt.imshow(imgTest,cmap='gray')

greyTest = imgTest/255
# greyTest = greyTest[:,:,2]
BWTest = greyTest < 0.75
BWTest = BWTest[:,:,2]

plt.figure()
plt.imshow(BWTest,cmap='gray')
plt.show()
 

# define normalized 1D gaussian
def gaus1d(x=0, mx=0, sx=1):
    return 1 / (2 * np.pi * sx) * np.exp(-((x - mx)**2 / (2 * sx**2) ))


# traning phase
mu = np.zeros((greyTrain.shape[2],2))
std = np.zeros((greyTrain.shape[2],2))
# mu = np.zeros((1,2))
# std = np.zeros((1,2))

for i in range(greyTrain.shape[2]):
    mu[i,0] = np.mean( greyTrain[:,:,i] * BWTrain)
    std[i,0] = np.std( greyTrain[:,:,i] * BWTrain)
    mu[i,1] = np.mean( greyTrain[:,:,i] * ~BWTrain)
    std[i,1] = np.std( greyTrain[:,:,i] * ~BWTrain)


# testing phase
pst = np.ones((imgTest.shape[0],imgTest.shape[1],greyTest.shape[2],2))
# pst = np.ones((imgTest.shape[0],imgTest.shape[1],1,2))


for i in range(greyTest.shape[2]):
# for i in range(1):
    for C in range(2):
        pst[:,:,i,C] = gaus1d( greyTest[:,:,i] , mu[i,C], std[i,C] )

pst = np.prod(pst,axis=2)
BW = np.float64(np.argmin(pst,2))

plt.figure()
plt.imshow(BW,cmap='gray')
plt.show()

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
# gray = rgb2gray(img)    
# plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
# plt.show()


# from IPython.display import display, Image
# display(Image(filename='img1.jpg'))

# from PIL import Image
# #read the image
# im = Image.open('img1.jpg')
# #show image
# im.show()

