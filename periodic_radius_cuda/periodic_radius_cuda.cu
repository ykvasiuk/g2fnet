#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "periodic_radius_cuda.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#define THREADS 256


// CUDA kernel definition
template <typename scalar_t>
__global__ void
radius_kernel_periodic(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
                       const int64_t *__restrict__ ptr_x,
                       const int64_t *__restrict__ ptr_y, int64_t *__restrict__ row,
                       int64_t *__restrict__ col, const scalar_t r, const int64_t n,
                       const int64_t m, const int64_t dim, const int64_t num_examples,
                       const int64_t max_num_neighbors, const scalar_t L) {

  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  int64_t count = 0;
  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t dist = 0;
    for (int64_t d = 0; d < dim; d++) {
      scalar_t diff = abs(x[n_x * dim + d] - y[n_y * dim + d]);
      scalar_t periodic_diff = min(diff, L - diff);  // Account for periodic boundary conditions
      dist += periodic_diff * periodic_diff;
    }

    if (dist < r) {
      row[n_y * max_num_neighbors + count] = n_y;
      col[n_y * max_num_neighbors + count] = n_x;
      count++;
    }

    if (count >= max_num_neighbors)
      break;
  }
}

torch::Tensor radius_periodic_cuda(const torch::Tensor x, const torch::Tensor y,
                                   torch::optional<torch::Tensor> ptr_x,
                                   torch::optional<torch::Tensor> ptr_y, const double r,
                                   const int64_t max_num_neighbors, const double L) {
  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CUDA(y);
  CHECK_CONTIGUOUS(y);
  CHECK_INPUT(y.dim() == 2);
  CHECK_INPUT(x.size(1) == y.size(1));

  cudaSetDevice(x.get_device());


  if (ptr_x.has_value()) {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  } else
    ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                          x.options().dtype(torch::kLong));

  if (ptr_y.has_value()) {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  } else
    ptr_y = torch::arange(0, y.size(0) + 1, y.size(0),
                          y.options().dtype(torch::kLong));

  CHECK_INPUT(ptr_x.value().numel() == ptr_y.value().numel());

  auto row =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());
  auto col =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());

  dim3 BLOCKS((y.size(0) + THREADS - 1) / THREADS);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = x.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "_", [&] {
        radius_kernel_periodic<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
            ptr_x.value().data_ptr<int64_t>(),
            ptr_y.value().data_ptr<int64_t>(), row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(), r * r, x.size(0), y.size(0), x.size(1),
            ptr_x.value().numel() - 1, max_num_neighbors, L);
      });

  auto mask = row != -1;
  return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}



// C++ interface (the function that will be called from Python)
torch::Tensor radius_periodic(const torch::Tensor x, const torch::Tensor y,
                              torch::optional<torch::Tensor> ptr_x,
                              torch::optional<torch::Tensor> ptr_y, const double r,
                              const int64_t max_num_neighbors, const double L) {
  return radius_periodic_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors, L);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radius_periodic", &radius_periodic, 
          "Radius with Periodic Boundary (CUDA)",
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("ptr_x") = pybind11::none(), 
          pybind11::arg("ptr_y") = pybind11::none(), pybind11::arg("r"), 
          pybind11::arg("max_num_neighbors"), pybind11::arg("L"));
}