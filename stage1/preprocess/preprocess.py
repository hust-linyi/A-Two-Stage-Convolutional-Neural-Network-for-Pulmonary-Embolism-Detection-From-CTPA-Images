import os
import numpy as np
import pydicom as dicom
import scipy.ndimage
from skimage import measure
from functools import partial
import warnings
import pandas
import SimpleITK as sitk

def load_itk_image(path):
    # itkimage = [sitk.ReadImage(path + '/' + s) for s in os.listdir(path) if s.endswith('.dcm')]
    # itkimage = sitk.ReadImage(path + '.dcm')
    series_IDs = sitk.ImageSeriesReader_GetGDCMSeriesIDs(path)
    series_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_filenames)
    itkimage = series_reader.Execute()

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpySpacing


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    #[z,y.x]
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]
    background_label1 = labels[0,511,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    binary_image[background_label1 == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def resample(image, spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def savenpy(id,annos,filelist,prep_folder,data_path,black_list):
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
    if name in black_list:
        print name + 'is in the blacklist'
        return 0
    try:
        if isClean:
            sliceim_pixels, spacing = load_itk_image(os.path.join(data_path, name))
            Mask = segment_lung_mask(sliceim_pixels, True)

            sliceim_resampled = resample(sliceim_pixels, spacing, [1,1,1])

            newshape = np.round(np.array(Mask.shape)*spacing/resolution)
            zz,yy,xx= np.where(Mask)
            print spacing
            # print xx.shape, yy.shape, zz.shape
            box = np.array([[np.min(zz),np.max(zz)],[np.min(yy),np.max(yy)],[np.min(xx),np.max(xx)]])
            box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            box = np.floor(box).astype('int')
            margin = 5
            extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
            extendbox = extendbox.astype('int')
            print extendbox

            sliceim1 = lumTrans(sliceim_resampled)
            sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
            bones = sliceim*extramask>bone_thresh
            bones = sliceim1 > bone_thresh
            sliceim1[bones] = pad_value
            sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                        extendbox[1,0]:extendbox[1,1],
                        extendbox[2,0]:extendbox[2,1]]
            sliceim = sliceim2[np.newaxis,...]
            np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
            np.save(os.path.join(prep_folder,name+'_spacing'),spacing)
            np.save(os.path.join(prep_folder, name+'_extendbox.npy'), extendbox)
        if islabel:
            this_annos = np.copy(annos[annos[:, 0] == (name + '.npy')])
            label = []

            if len(this_annos)>0:
                for c in this_annos:
                    pos = c[1:4][::-1]
                    diameter_mm = np.sqrt(np.sum((c[4:]*spacing[::-1])**2))
                    label.append(np.concatenate([pos, [diameter_mm]]))

            label = np.array(label)
            if len(label)==0:
                label2 = np.array([[0,0,0,0]])
            else:
                label2 = np.copy(label).T
                label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
                label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
                label2 = label2[:4].T
            np.save(os.path.join(prep_folder,name+'_label.npy'), label2)
    except:
        print('bug in '+name)
        raise
    print(name+' done')


def full_prep(data_path,annos,prep_folder,black_list):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)


    print('starting preprocessing')
    pool = Pool()
    filelist = [f for f in os.listdir(data_path)]
    filelist.sort()
    partial_savenpy = partial(savenpy,annos=annos,filelist=filelist,prep_folder=prep_folder,
                              data_path=data_path)
    
    N = len(filelist)
    print N
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist

def preprocess_pe():
    data_path = ''
    prep_folder = ''
    luna_label = ''
    black_list = []
    annos = np.array(pandas.read_csv(luna_label))
    full_prep(data_path=data_path, annos=annos,prep_folder=prep_folder,black_list=black_list)

if __name__ == '__main__':
    preprocess_pe()
