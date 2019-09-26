import numpy as np
# import cv2
import os
from skimage.measure import label
from skimage.measure import regionprops
import csv


def get_coordinates(label_path):
    label_image = np.load(label_path)
    label_img = label(label_image, connectivity=label_image.ndim)
    props = regionprops(label_img)

    coordinates = []
    center = []

    for n, prop in enumerate(props):
        coordinate = []
        coordinate.append(prop.bbox[0])  # ymin
        coordinate.append(prop.bbox[1])  # xmin
        coordinate.append(prop.bbox[2])  # zmin
        coordinate.append(prop.bbox[3])  # ymax
        coordinate.append(prop.bbox[4])  # xmax
        coordinate.append(prop.bbox[5])  # zmax
        # only keep the label z > 10
#        if prop.bbox[5] - prop.bbox[2] > 5:
#            center.append(prop.centroid)
#
#            coordinates.append(coordinate)

    new_center, new_coordinates = center, coordinates
    return new_center, new_coordinates

if __name__ == '__main__':
    mask_numpy_path = ''
    save_csv_path = ''
    mask_names = os.listdir(mask_numpy_path)
    mask_names.sort()

    # get csv [seriesuid, x, y, z, diameter_mm]
    with open(save_csv_path + '/3D_label_hospital_new_final_12.6_2.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
        for mask_name in mask_names:
            center, bbox = get_coordinates(os.path.join(mask_numpy_path, mask_name))
            for i in range(len(center)):
                diameter_mm = np.sqrt(np.square(bbox[i][3] - bbox[i][0])
                                      + np.square(bbox[i][4] - bbox[i][1]) + np.square(bbox[i][5] - bbox[i][2]))
                writer.writerow([mask_name, center[i][1], center[i][0], center[i][2], diameter_mm])

