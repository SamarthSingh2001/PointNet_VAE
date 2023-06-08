# PointNet.pytorch VAE version
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

Added a VAE training file for generative AI purposes

# Download data and running

```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool
```
cd scripts
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training 
```
cd utils
python train_VAE.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type shapenet
```

Use `--feature_transform` to use feature transform.

# Performance

## VAE performance

Checkout the presentation file. More work to be done. For improvement, will implement segmentation algorithm into the VAE.

Still figuring out the display.

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
# PointNet_VAE
