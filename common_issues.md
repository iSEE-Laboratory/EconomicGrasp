# Common Issues

## Installation Failure 1

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

## Installation Failure 2

Sometimes we may meet anthor bug `fatal error: cusolverDn.h: No such file or directory` when install pointnet2. Just add `export PATH=/usr/local/cuda/bin:$PATH` before installation.
