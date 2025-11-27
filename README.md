# **1.Environment**</br>
following the steps below (python=3.8):
```
pip install causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl # We have directly provided it in the 'environment' directory, and it can be downloaded directly.
pip install transformers==4.45.1      triton==3.0.0 
pip install mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl # # We have directly provided it in the 'environment' directory, and it can be downloaded directly.
pip install torch==1.13.0 torchvision==0.9.1+cu111
pip install monai==1.3.2
pip install timm tensorboardX  scikit-image torchstat torchsummary
```

# **2. Datasets** </br>
*A. Dataset link* </br>
Download the RIM-ONE-r3 dataset from [this](https://www.idiap.ch/software/bob/docs/bob/bob.db.rimoner3/master/) link. </br>
Download the RIM-ONE DL dataset from [this](https://github.com/miag-ull/rim-one-dl) link. </br>
Download the Drishti-GS dataset from [this](https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation) link. </br>
Download the REFUGE  dataset from [this](https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge#files) link. </br>

*B. Dataset folders* </br>
Take Drishti-GS for example: Split the downloaded dataset into training, test, and validation sets with a 7:1:2 ratio, and place them at the following locations.
```
../dataset/fundus/gdrishtiGS/train/ROIs/...
../dataset/fundus/gdrishtiGS/test/ROIs/...
../dataset/fundus/gdrishtiGS/val/ROIs/...
```
# **3. FastSCVM** </br>
Before starting training, you can directly run FastSCVM3_New.py to verify that the environment is correctly installed — the terminal will immediately output the number of parameters and computational cost (FLOPs) of FastSCVM.
```
python3 FastSCVM3_New.py
```
# **4. Train** </br>
In line 52 of train.py, modify it to the dataset you want to train: gdrishtiGS, refuge, RIM-ONE-r3, or N_G (where N_G represents RIM-ONE_DL). In line 90 of train.py, modify it to the local path of your own dataset.And then simply run:
```
python3 train.py
```
# **5. Test** </br>
After training is completed, the logs directory will retain the weights and the performance of each weight on the validation set. Use the best weights to run test2.py and check its performance on the test set.
The areas that may need to be modified in test2.py are as follows:
```
In line 40, modify the location of the best weight
in line 44, modify the dataset used
in lines 52 and 110, modify the image size to match the training set
in line 56, modify the dataset location
in line 64, modify the location of the test set mask
in line 275, modify the location where the predicted images will be saved.
```
And then simply run:
```
python3 test2.py
```
After running test2.py, all the evaluation metrics covered in this paper will be output in the terminal. The predicted segmentation masks will be saved in the corresponding directories under results
