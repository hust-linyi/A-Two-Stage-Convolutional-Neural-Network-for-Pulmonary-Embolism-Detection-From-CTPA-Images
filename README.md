# A Two Stage Convolutional Neural Network for Pulmonary Embolism Detection From CTPA Images
By Xin Yang, Yi Lin, Jianchao Su, Xiang Wang, Xiang Li, Jingen Lin, Kwang-Ting Cheng
### Introduction
This is the official repo for "A Two Stage Convolutional Neural Network for Pulmonary Embolism Detection From CTPA Images". For more details please refer to our paper. Please cite the [paper](https://ieeexplore.ieee.org/abstract/document/8746218/) in your publications if you find the source code useful to your research.
### Citing Our Paper

    @article{yang2019two,
      title={A Two-Stage Convolutional Neural Network for Pulmonary Embolism Detection From CTPA Images},
      author={Yang, Xin and Lin, Yi and Su, Jianchao and Wang, Xiang and Li, Xiang and Lin, Jingen and Cheng, Kwang-Ting},
      journal={IEEE Access},
      volume={7},
      pages={84849--84857},
      year={2019},
      publisher={IEEE}
    }
### Requirements

    python 2.7
    pytorch >= 0.4.0
    
### Usage
#### Clone the repository
        
        $ git clone git@github.com:hust-linyi/A-Two-Stage-Convolutional-Neural-Network-for-Pulmonary-Embolism-Detection-From-CTPA-Images.git

#### Preparation:

  Download the [FUMPE dataset](https://figshare.com/collections/FUMPE/4107803/1)
   
#### For stage 1:

        cd stage1

#### Train

0. Generate *.csv file for groundtruth, refer to ./preprocess/get_3D_label.py. For preprocess, please refer to ./preprocess/preprocess.py

1. In ./stage1/detector/ folder and run:

        train.sh
        
#### Test

In ./detector/ folder and run:

        test.sh
        
#### For stage 2:

        cd stage2

#### Train

0. Generate new *.csv file for groundtruth, refer to ./prepare_csv.py

        python classification.py --test 0
       
#### Test
        
        python classification.py --test 1

### Notes

For the consideration of patient privacy, we did not release the PE129 dataset.
