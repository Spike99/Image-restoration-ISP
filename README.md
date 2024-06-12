# Image-restoration-ISP
# Low-light-Image-Enhancement
MindSpore implementation of low-light stereo image enhancement methods
## Installation
This implementation runs on python >= 3.7, use pip to install dependencies:
```
conda install mindspore=2.2.14 -c mindspore -c conda-forge
```

## Usage
Use the `test.py` script to enhance your images.
```
usage: test.py [--data_source] [--experiment] [--train_bs_per_gpu] [--val_gap] [-ul] [-s SIGMA]
               [-bc BC] [-bs BS] [-be BE] [-eps EPS]

optional arguments:
  --data_source         Your dir for dataset.
  --experiment          The version of this experiment.
  --train_bs_per_gpu    The batch size executed on each GPU.
  --val_gap             Specify several inference processes to calculate a PSNR.
  --print_gap           Specify several inference processes to display current information.
  --seed                The number of seeds specified by the experimenter.
  --exp_name            Specify the name of the experimental dataset. This project has predefined datasets such as Flickr1024 and Middlebury.
  --pretrained          Address of model parameters.
```

### Example
```
python test.py --data_source /jizhicheng/Data/DataSets/Ideas/Stereo/iPASSR/test/Flickr1024 --experiment test_0_1 --val_gap 1 --print_gap 86 --seed 007 --exp_name  Flickr1024_psnr --pretrained ../results/test_0_1/models/optimal_psnr.onnx
```
