// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_IMGPROC_POLY_TO_MASK_H_
#define DALI_KERNELS_IMGPROC_POLY_TO_MASK_H_

#include <cuda_runtime.h>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/core/static_switch.h"

namespace dali {

namespace kernels {

namespace detail {
 
template <typename T>
__global__ void PolyToMaskKernel(const int* masks_meta,
                                 const int* sample_mask_meta_indices,
                                 const int* polys_meta, T* out_masks, int N) {
  int sample_idx = blockIdx.x;
  int mask_idx = blockIdx.y;

  for (; sample_idx < N; N += blockDim.x) {
    const auto* sample_masks_meta = sample_mask_meta_indices[sample_idx];
    int n
  }
}

}  // namespace detail

template <typename Type>
class DLL_PUBLIC PolyToMaskKernel {
 public:
  DLL_PUBLIC PolyToMaskKernel() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InTensorCPU<Type, 4> &in) {
    KernelRequirements req;
    req.output_shapes = {TensorListShape<DynamicDimensions>({in.shape})};
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &Context, OutTensorCPU<Type, 4> &out,
      const InTensorGPU<Type, 4> &in, masks_meta, polys_meta) {
    auto num_samples = static_cast<size_t>(in.num_samples());
    DALI_ENFORCE(flip_x.size() == num_samples && flip_y.size() == num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      auto layers = in.tensor_shape(i)[0];
      auto height = in.tensor_shape(i)[1];
      auto width = in.tensor_shape(i)[2];
      auto channels = in.tensor_shape(i)[3];
      auto in_data = in[i].data;
      auto out_data = out[i].data;
      detail::gpu::FlipImpl(out_data, in_data, layers, height, width, channels,
          flip_z[i], flip_y[i], flip_x[i], context.gpu.stream);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_POLY_TO_MASK_H_
