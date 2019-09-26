import numpy as np
import matplotlib.pyplot as plt
import os

mask_numpy_path = '/home/linyi/Code/pytorch/pe_detection_clean/Data/challenge/preprocess/'
filenames = [t for t in os.listdir(mask_numpy_path) if 'clean' in t]
filenames.sort()
# for i in range(len(filenames)):
for i in range(1):
    # filename = filenames[i]
    # filename = 'HHCT200045878_clean.npy'
    # filename = 'HHCT200043845_clean.npy'
    # filename = 'HHCT200049011_clean.npy'
    filename = '029' + '_clean.npy'
    print i
    print filename
    mask = np.load(mask_numpy_path + filename)
    # print mask.shape
    mask = mask.squeeze(0)
    mask = mask.transpose(1, 2, 0)
    print mask.shape

    mask_slice0 = mask[:,:, 0]
    mask_slice1 = mask[:,:, 1]
    # mask_slice2 = mask[:,:, 59]
    mask_slice2 = mask[:,:, 2]
    mask_slice3 = mask[:,:, 100]

    # pix = mask[248, 196, 117]
    figure = plt.figure(20*20)
    ax1 = figure.add_subplot(221)
    ax1.imshow(mask_slice0, cmap=plt.cm.get_cmap('gray'))
    ax2 = figure.add_subplot(222)
    ax2.imshow(mask_slice1, cmap=plt.cm.get_cmap('gray'))
    ax3 = figure.add_subplot(223)
    ax3.imshow(mask_slice2, cmap=plt.cm.get_cmap('gray'))
    ax4 = figure.add_subplot(224)
    ax4.imshow(mask_slice3, cmap=plt.cm.get_cmap('gray'))

    plt.show()
