import numpy as np
import os
import csv

def real_box(datapath):
    filenames = os.listdir(datapath)
    filenames.sort()
    f = open(save_dir + 'box1.csv', 'w')
    fwriter = csv.writer(f)
    fwriter.writerow(["seriesuid","xmin","xmax","ymin","ymax","zmin", "zmax"])
    for filename in filenames:
        if 'extendbox' in filename:
            coords = np.load(os.path.join(datapath, filename))
            fwriter.writerow([filename.replace('_extendbox.npy', ''), coords[2][0], coords[2][1],
                              coords[1][0], coords[1][1], coords[0][0], coords[0][1]])

if __name__=='__main__':
    datapath = '/home/linyi/Code/pytorch/pe_detection_clean/Data/challenge/preprocess/'
    save_dir = './temp/'
    real_box(datapath)
