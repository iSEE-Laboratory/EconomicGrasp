#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    long *idx_dev = idx.data<long>();

  if (ref.type().is_cuda()) {
#ifdef WITH_CUDA
    // 分配内存并获取数据指针
    at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    at::Tensor tensor = at::empty({ref_nb * query_nb}, options);
    float* dist_dev = tensor.data_ptr<float>();    
    for (int b = 0; b < batch; b++)
    {
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }
    // 释放内存
    tensor = at::Tensor();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("error in knn: %s\n", cudaGetErrorString(err));
    //     // THError("aborting");
    // }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}


