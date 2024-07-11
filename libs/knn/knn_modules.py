import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
from knn_pytorch import knn_pytorch
# import knn_pytorch


def knn(ref, query, k=1):
  """ Compute k nearest neighbors for each query point.
  """
  device = ref.device
  ref = ref.float().to(device)
  query = query.float().to(device)
  inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
  knn_pytorch.knn(ref, query, inds)
  return inds


if __name__ == "__main__":
  seed = 0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  ref = torch.randn(1, 100, 100)  # (batch, dim, n)
  query = torch.randn(1, 100, 10)  # (batch, dim, n)
  ref.to("cuda")
  query.to("cuda")
  print(knn(ref, query))