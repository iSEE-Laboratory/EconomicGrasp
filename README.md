# EconomicGrasp

Official implement of SparseGrasp

Xiao-Ming Wu, Wei-Shi Zheng*.

If you have any questions, feel free to contact me by wuxm65@mail2.sysu.edu.cn.


## Installation
The installation process is a little complex, please follow the order to ensure success. Some common issues are also shown below.

### MinkowskiEngine
Install MinkowskiEngine following the Anaconda installation in https://github.com/NVIDIA/MinkowskiEngine.
About the pytorch version, we should download the pytorch match your cuda driver (torch.cuda.is_available() = True).
```bash
conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### Pip Dependency
Install dependent packages via Pip.
```bash
pip install -r requirements.txt
```

### PointNet2
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```

### KNN 
KNN is use for dataset generation. Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```

### PyTorch3D
Download the PyTorch3D version meet your PyTorch version from [PyTorch3D](https://github.com/facebookresearch/pytorch3d/releases).
```bash
tar -zxvf pytorch3d-0.7.2.tar.gz
cd pytorch3d-0.7.2/
python setup.py install
```

### graspnetAPI 
GraspnetAPI is used for dataset generation. Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

## Graspness Generation
Point level graspness label are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_graspness.py](dataset/generate_graspness.py).
```bash
cd dataset
python generate_graspness.py --dataset_root /data/graspnet --camera_type kinect
```

## Training and Testing
Waiting.

## Common Issues

### Install Fail 1
Sometimes we may meet bug  `fatal error: pybind11/pybind11.h: No such file or directory`, we should install it using
```bash
sudo apt-get install python3-dev
sudo apt-get install cmake
pip install pytest numpy scipy pybind11
git clone https://github.com/pybind/pybind11.git
cd pybind11
sudo mkdir build
cd build
sudo cmake ..
sudo make check -j 4
sudo make install
```

### Install Fail 2
Sometimes we may meet anthor bug `fatal error: cusolverDn.h: No such file or directory` when install pointnet2. Just add `export PATH=/usr/local/cuda/bin:$PATH` before installation.



### Training Failure Case
When training failure, it will be easy to cause two situations: no mask points in graspness or all the points are masked out in graspness.

When no mask points, it will occur `RuntimeError: CUDA error: an illegal memory access was encountered.` in the mask line. 

If all the points are masked out, it will occur it will occur `RuntimeError: CUDA error: an illegal memory access was encountered.` in the pointnet library.


## Results
Waiting.


## Citation
Waiting.
```
