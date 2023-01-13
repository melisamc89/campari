
import numpy as np
import czifile as czifile
import matplotlib.pylab as plt
import skimage as sk
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from skimage.io import imread

from skimage.morphology import binary_dilation
from skimage.filters import gaussian
from skimage import measure

from skimage.draw import polygon

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

figure_path = '/home/melisamc/Documentos/campari/figures/'

file_path = '/home/melisamc/Documentos/campari/data/example_0/'
file_name1 = 'Snap-3085.czi'
file_name2 = 'Snap-3086.czi'

images2 = czifile.imread(file_path + file_name1)
images1 = czifile.imread(file_path + file_name2)
img = images1[0,:,:,0]#[0,500:600,600:700,0]
images1 = img.copy()


figure, axes = plt.subplots(1,1)
axes.imshow(images1[:,:], cmap = 'gray')
axes.set_title('Red')
plt.show()

f, ax = plt.subplots()
_ = ax.hist(img.ravel(),color='k',bins=100)
ax.set_yscale('log')
ax.set_xlabel('pixel')
ax.set_ylabel('# of pixels')
plt.show()

threshold = 2000
mask_image = np.zeros(img.shape)
mask_image[img>threshold] = 255
f,ax = plt.subplots()
ax.imshow(mask_image, cmap='Greys')
plt.show()

new_mask = gaussian(mask_image, sigma=2)
f,ax = plt.subplots()
ax.imshow(new_mask, cmap='Greys')
plt.show()

f,ax = plt.subplots()
contours = measure.find_contours(new_mask, level=125 ) # level is half of 255 (ish). What happens if we change it?
ax.imshow(new_mask, cmap='Greys')
for contour in contours:
  ax.plot(contour[:,1],contour[:,0],color='r')
plt.show()

perimeter = np.zeros((len(contours,)))
i = 0
for contour in contours:
  perimeter[i] = len(contour)
  i+=1

threshold_perim = 50
perimeter_boolean =  np.zeros((len(contours,)))
for i in range(len(contours)):
  if perimeter[i] >threshold_perim:
    perimeter_boolean[i] = 1

NBINS = 10
f,ax = plt.subplots(2,2)
ax[0,0].imshow(new_mask, cmap='Greys')
ax[0,0].set_title('Intensity Threshold:' + str(threshold))
for contour in contours:
  ax[0,0].plot(contour[:,1],contour[:,0],color='r')
ax[0,0].set_xticklabels([])
ax[0,0].set_yticklabels([])
ax[0,1].set_title('Perimeter Threshold:' + str(threshold_perim))
ax[0,1].imshow(new_mask, cmap='Greys')
i = 0
for contour in contours:
  if perimeter_boolean[i] == 1:
    ax[0,1].plot(contour[:, 1], contour[:, 0], color='b')
  i+=1
ax[0,1].set_xticklabels([])
ax[0,1].set_yticklabels([])
_ = ax[1,0].hist(img.ravel(),color='r',bins=100)
ax[1,0].set_yscale('log')
ax[1,0].set_xlabel('pixel')
ax[1,0].set_ylabel('#')
#ax[1,0].set_ylim([0,1])
ax[1,1].hist(perimeter,color = 'b')
ax[1,1].set_xlim([0,250])
ax[1,1].set_xlabel('Perimeter')
ax[1,1].set_ylabel('# ')
#ax[1,1].set_ylim([0,1])
plt.show()
f.savefig(figure_path + 'First_step.png')

L = 25
i = 0
for contour in contours:
  if perimeter_boolean[i] == 1:
    print(i)
    f, ax = plt.subplots(1, 2)
    cm = np.mean(contour,axis = 0)
    x0 = np.max([0, int(cm[0]) - L])
    y0 = np.max([0, int(cm[1]) - L])
    x1 = np.min([1024, int(cm[0]) + L])
    y1 = np.min([1024, int(cm[1]) + L])
    image_data = images1[x0:x1, y0:y1]
    ax[0].imshow(image_data, cmap='Greys')
    ax[0].set_title('Intensity Threshold:' + str(threshold), fontsize = 15)
    ax[0].plot(contour[:, 1] - cm[1] + L, contour[:, 0] - cm[0]+L, color='b')
    # ax[0].set_xticklabels([])
    # ax[0].set_yticklabels([])
    _ = ax[1].hist(image_data.ravel(),color='r',bins=20)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('pixel')
    ax[1].set_ylabel('#')
    ax[1].set_title('Cell Histogram', fontsize = 15)
    f.set_size_inches([10,5])
    f.savefig(figure_path + 'Single_cell_'+str(i)+'.png')
    plt.show()
    plt.close()
  i = i+1

#
# i = 46
# for contour in contours:
#   if perimeter_boolean[i] == 1:
#
# cm = np.mean(contour,axis = 0)
# image_data = images1[int(cm[0])-L:int(cm[0])+L,int(cm[1])-L:int(cm[1])+L]
#
# threshold = 3000
# mask_image = np.zeros(image_data.shape)
# mask_image[image_data>threshold] = 255
# f,ax = plt.subplots()
# ax.imshow(mask_image, cmap='Greys')
# plt.show()
#
# new_mask = gaussian(mask_image, sigma=1)
# f,ax = plt.subplots()
# ax.imshow(new_mask, cmap='Greys')
# plt.show()
#
# f,ax = plt.subplots()
# contours_1 = measure.find_contours(new_mask, level=125 ) # level is half of 255 (ish). What happens if we change it?
# ax.imshow(new_mask, cmap='Greys')
# for contour_1 in contours_1:
#   ax.plot(contour_1[:, 1], contour_1[:, 0] , color='b')
# plt.show()
#
# cm = np.mean(contour,axis = 0)
# image_data = images1[int(cm[0])-L:int(cm[0])+L,int(cm[1])-L:int(cm[1])+L]
#
# threshold = 5000
# mask_image = np.zeros(image_data.shape)
# mask_image[image_data>threshold] = 255
# f,ax = plt.subplots()
# ax.imshow(mask_image, cmap='Greys')
# plt.show()
#
# new_mask = gaussian(mask_image, sigma=1)
# f,ax = plt.subplots()
# ax.imshow(new_mask, cmap='Greys')
# plt.show()
#
# f,ax = plt.subplots()
# contours_1 = measure.find_contours(new_mask, level=125 ) # level is half of 255 (ish). What happens if we change it?
# ax.imshow(new_mask, cmap='Greys')
# for contour_1 in contours_1:
#   ax.plot(contour_1[:, 1], contour_1[:, 0] , color='b')
# plt.show()
#
# j = 0
# for contour_1 in contours_1:
#   f, ax = plt.subplots(1, 2)
#   x0 = np.max([0, int(cm[0]) - L])
#   y0 = np.max([0, int(cm[1]) - L])
#   x1 = np.min([1024, int(cm[0]) + L])
#   y1 = np.min([1024, int(cm[1]) + L])
#
#   cm_1 = np.mean(contour_1,axis = 0)
#
#   x0_1 = np.max([0, int(cm_1[0]) - L])
#   y0_1 = np.max([0, int(cm_1[1]) - L])
#   x1_1 = np.min([50, int(cm_1[0]) + L])
#   y1_1 = np.min([50, int(cm_1[1]) + L])
#
#   image_data_1 = image_data[x0_1:x1_1,y0_1:y1_1]
#
#   ax[0].imshow(image_data_1, cmap='Greys')
#   ax[0].set_title('Intensity Threshold:' + str(threshold), fontsize = 15)
#   ax[0].plot(contour_1[:, 1] - y0_1, contour_1[:, 0] - x0_1, color='b')
#
#   # ax[0].set_xticklabels([])
#   # ax[0].set_yticklabels([])
#   _ = ax[1].hist(image_data_1.ravel(),color='r',bins=20)
#   ax[1].set_yscale('log')
#   ax[1].set_xlabel('pixel')
#   ax[1].set_ylabel('#')
#   ax[1].set_title('Cell Histogram', fontsize = 15)
#   f.set_size_inches([10,5])
#   f.savefig(figure_path + 'Single_cell_i'+str(i)+'_j_'+str(j)+'.png')
#   plt.show()
#   plt.close()
#   j = j +1
# i = i+1


threshold = 2000
L = 25
i = 0
for contour in contours:
  if perimeter_boolean[i] == 1:
    print(i)
    f, ax = plt.subplots(2, 2)
    cm = np.mean(contour,axis = 0)
    x0 = np.max([0, int(cm[0]) - L])
    y0 = np.max([0, int(cm[1]) - L])
    x1 = np.min([1024, int(cm[0]) + L])
    y1 = np.min([1024, int(cm[1]) + L])
    image_data = images1[x0:x1, y0:y1]

    mask_image1 = np.zeros(image_data.shape)
    mask_image = np.zeros(image_data.shape)
    mask_image1[image_data > threshold] = 255
    mask_image[mask_image1 > 20] = 255
    new_mask = gaussian(mask_image, sigma=3)
    contours_ = measure.find_contours(new_mask, level=125, fully_connected='high')

    ax[1,0].imshow(new_mask, cmap='Greys')
    ax[1,0].set_title('2nd Intensity Threshold:' + str(20), fontsize = 15)

    contours_connected = np.vstack((contours_))
    for contour__ in contours_:
      ax[1,0].plot(contour__[:, 1], contour__[:, 0], 'r')
    contours_connected = np.vstack((contours_connected[-1, :], contours_connected))
    ax[1,0].plot(contours_connected[:, 1], contours_connected[:, 0], 'g')

    ax[0,0].imshow(image_data, cmap='Greys')
    ax[0,0].set_title('Intensity Threshold:' + str(threshold), fontsize = 15)
    ax[0,0].plot(contour[:, 1] - cm[1] + L, contour[:, 0] - cm[0]+L, color='b')
    # ax[0].set_xticklabels([])
    # ax[0].set_yticklabels([])
    _ = ax[0,1].hist(image_data.ravel(),color='r',bins=20)
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('pixel')
    ax[0,1].set_ylabel('#')
    ax[0,1].set_title('Image Histogram', fontsize = 15)

    _ = ax[1, 1].hist(new_mask.ravel(), color='r', bins=20)
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel('pixel')
    ax[1, 1].set_ylabel('#')
    ax[1, 1].set_title('Filtered Image Histogram', fontsize=15)
    f.set_size_inches([10,10])
    f.savefig(figure_path + 'Single_cell_'+str(i)+'_2.png')
    plt.show()
    plt.close()
  i = i+1

mask_image = np.zeros(img.shape)
mask_image[img>threshold] = 255
new_mask = gaussian(mask_image, sigma=4)
contours = measure.find_contours(new_mask, level=125, fully_connected='high')


# make a new mask
watershed_start = np.zeros(img.shape)
rr, cc = polygon(contours_connected[:,0], contours_connected[:,1])
watershed_start[rr,cc] = 1
f,ax = plt.subplots()
ax.imshow(watershed_start, cmap='Greys')
plt.show()


# apply watershed
distance = ndi.distance_transform_edt(watershed_start) #compute the distance image
coords = peak_local_max(distance, min_distance=5)#, labels=watershed_start) #use the distance image to find local maxima
_,inds = np.unique(distance[coords[:,0],coords[:,1]],return_index=True) #make sure they are unique
coords = coords[inds,:]
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True # make an image with 1's where local maxima are
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=watershed_start, watershed_line=True) # perform watershed


f,ax = plt.subplots(2,3, figsize=(8,8))
ax[0,0].imshow(images1[:,:], cmap='gray')
ax[0,1].imshow(mask_image, cmap='Greys')
ax[0,2].imshow(img, cmap='Greys')
ax[0,2].plot(contours_connected[:,1],contours_connected[:,0],'g')
ax[1,0].imshow(img, cmap='Greys')
ax[1,0].scatter(coords[:,1],coords[:,0],c='r')
ax[1,1].imshow(distance, cmap='Greys')
ax[1,1].scatter(coords[:,1],coords[:,0],c='r')
ax[1,2].imshow(labels, cmap='Greys')
f.tight_layout()
plt.show()