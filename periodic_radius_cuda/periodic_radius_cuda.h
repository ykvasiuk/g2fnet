#pragma once
#include "./extensions.h"

torch::Tensor radius_periodic_cuda(torch::Tensor x, torch::Tensor y,
                          torch::optional<torch::Tensor> ptr_x,
                          torch::optional<torch::Tensor> ptr_y, double r,
                          int64_t max_num_neighbors, double L); 
                          
