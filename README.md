# An Economic Framework for 6-DoF Grasp Detection

Official implement of EconomicGrasp | [Paper](https://arxiv.org/abs/2407.08366) | [Personal Homepage](https://dravenalg.github.io/).

**Xiao-Ming Wu&**, Jia-Feng Cai&, Jian-Jian Jiang, Dian Zheng, Yi-Lin Wei, Wei-Shi Zheng*

Accepted at ECCV 2024!

Super fast to converge, use it and you will like it!

If you have any questions, feel free to contact me by wuxm65@mail2.sysu.edu.cn.

## Abstract

 Robotic grasping in clutters is a fundamental task in robotic manipulation. In this work, we propose an economic framework for 6-DoF grasp detection, aiming to economize the resource cost in training and meanwhile maintain effective grasp performance. To begin with, we discover that the dense supervision is the bottleneck of current SOTA methods that severely encumbers the entire training overload, meanwhile making the training difficult to converge. To solve the above problem, we first propose an economic supervision paradigm for efficient and effective grasping. This paradigm includes a well-designed supervision selection strategy, selecting key labels basically without ambiguity, and an economic pipeline to enable the training after selection. Furthermore, benefit from the economic supervision, we can focus on a specific grasp, and thus we devise a focal representation module, which comprises an interactive grasp head and a composite score estimation to generate the specific grasp more accurately. Combining all together, the **EconomicGrasp** framework is proposed. Our extensive experiments show that EconomicGrasp surpasses the SOTA grasp method by about **3AP** on average, and with extremely low resource cost, for about **1/4** training time cost, **1/8** memory cost and 1/30 storage cost. Our code is available at \url{https://github.com/iSEE-Laboratory/EconomicGrasp}.

## Overall

<img src="imgs/framework.png" alt="model_framework" style="zoom:50%;" />

## How to Run

### Dependencies Installation

Please follow the instructions to ensure successful installation. We also write a [file](common_issues.md) to describe some common issues of the installation.

#### PyTorch & MinkowskiEngine

Install PyTorch. 

```bash
conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia 
# the version of pytorch should suit your cuda
# torch.cuda.is_available() == True, check the right CUDA version
```

Install the MinkowskiEngine.

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### Pip Dependency

Install dependent packages via Pip.

```bash
pip install -r requirements.txt
```

#### PointNet2

Compile and install pointnet2 operators.

```bash
cd pointnet2
python setup.py install
```

#### KNN 

Compile and install knn operator.

```bash
cd knn
python setup.py install
```

#### GraspNetAPI 

Install graspnetAPI for evaluation.

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

### Graspness Generation

Generate graspness. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). 
```bash
cd dataset
python generate_graspness.py --dataset_root /home/xiaoming/dataset/graspnet --camera_type kinect
```

### Sparse Dataset Generation

After generating the graspness, we can generate our economic supervision.

```bash
cd dataset
python generate_economic.py --dataset_root /home/xiaoming/dataset/graspnet --camera_type kinect
```

### Training

Then we can train our model.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model economicgrasp --camera kinect --log_dir results/economicgrasp --max_epoch 10 --batch_size 4 --dataset_root /home/xiaoming/dataset/graspnet
```

### Testing

[EconomicGrasp-kinect](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v0.1/economicgrasp_kinect.tar)

[EconomicGrasp-realsense](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v0.1/economicgrasp_realsense.tar)

For testing, there are seen, similar, novel settings.

```bash
# seen
CUDA_VISIBLE_DEVICES=0 python test.py --model economicgrasp --save_dir results/economicgrasp/test_ep10_seen --checkpoint_path results/economicgrasp/economicgrasp_epoch10.tar --camera kinect --dataset_root /home/xiaoming/dataset/graspnet --test_mode seen --inference
# similar
CUDA_VISIBLE_DEVICES=1 python test.py --model economicgrasp --save_dir results/economicgrasp/test_ep10_similar --checkpoint_path results/economicgrasp/economicgrasp_epoch10.tar --camera kinect --dataset_root /home/xiaoming/dataset/graspnet --test_mode similar --inference
# novel
CUDA_VISIBLE_DEVICES=2 python test.py --model economicgrasp --save_dir results/economicgrasp/test_ep10_novel --checkpoint_path results/economicgrasp/economicgrasp_epoch10.tar --camera kinect --dataset_root /home/xiaoming/dataset/graspnet --test_mode novel --inference

```

## Results

Grasp performance in GraspNet-1Billion dataset.

| Camera    | Seen (AP) | Similar (AP) | Novel (AP) |
| --------- | --------- | ------------ | ---------- |
| Kinect    | 62.59     | 51.73        | 19.54      |
| Realsense | 68.21     | 61.19        | 25.48      |

We also test the training time, memory cost in an empty machine with one RTX 3090 GPU. 

| Time  | Main Memory | GPU memory |
| ----- | ----------- | ---------- |
| 8.3 h | 4.2 G       | 5.81 G     |

NOTE1: We have already printed the remaining time in our log output and you can see it when you run the code in your machine.

NOTE2: When you use the model at the first time, it will not be as fast as you think. It takes maybe 60ep to warmup at the first time, and then next time your training will be normal and fast.

## Citation

```
@misc{wu2024economicframework6dofgrasp,
      title={An Economic Framework for 6-DoF Grasp Detection}, 
      author={Xiao-Ming Wu and Jia-Feng Cai and Jian-Jian Jiang and Dian Zheng and Yi-Lin 			Wei and Wei-Shi Zheng},
      year={2024},
      eprint={2407.08366},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.08366}, 
}
