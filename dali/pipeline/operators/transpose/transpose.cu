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

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/transpose/transpose.h"
#include "dali/error_handling.h"
#include "dali/kernels/static_switch.h"

namespace dali {

#define cuttCheck(stmt) do {                                   \
  cuttResult err = stmt;                                       \
  if (err != CUTT_SUCCESS) {                                   \
    DALI_FAIL("Error while transposing " + std::string(#stmt)); \
  }                                                            \
} while (0)

std::pair<IntArr, IntArr>
RowToColumnMajor(const int *dims, const int *perm, int rank) {
  IntArr new_dims(new int[rank]);
  IntArr new_perm(new int[rank]);

  for (int i = 0; i < rank; ++i) {
    new_dims[i] = dims[rank - 1 - i];
    new_perm[i] = rank - 1 - perm[rank - 1 - i];
  }
  return {std::move(new_dims), std::move(new_perm)};
}

template <typename T = double>
void cuttSanityCheck(int *dims, int *perm, int rank) {
  int vol = 1;
  for (int i = 0; i < rank; ++i) {
    vol *= dims[i];
  }
  T *datain = new T[vol];
  for (int i = 0; i < vol; ++i) datain[i] = static_cast<T>(i);

  T *dataout = new T[vol];

  T *datagpuin;
  T *datagpuout;

  cudaMalloc(&datagpuin, vol * sizeof(T));
  cudaMalloc(&datagpuout, vol * sizeof(T));

  cudaMemcpy(datagpuin, datain, vol * sizeof(T), cudaMemcpyHostToDevice);

  IntArr c_dims, c_perm;
  std::tie(c_dims, c_perm) = RowToColumnMajor(dims, perm, rank);

  cuttHandle plan;
  cuttCheck(cuttPlan(&plan, rank, c_dims.get(), c_perm.get(), sizeof(T), 0));
  cuttCheck(cuttExecute(plan, datagpuin, datagpuout));

  CUDA_CALL(cudaStreamSynchronize(0));
  cudaMemcpy(dataout, datagpuout, vol * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < vol; ++i)
    std::cout << dataout[i] << " ";
  std::cout << std::endl;
  cuttCheck(cuttDestroy(plan));
  cudaFree(datagpuin);
  cudaFree(datagpuout);
}

namespace kernel {

template <typename T>
void cuTTKernel(const TensorList<GPUBackend>& input,
                TensorList<GPUBackend>& output,
                const std::vector<int>& permutation,
                cudaStream_t stream) {
  int batch_size = static_cast<int>(input.ntensor());
  for (int i = 0; i < batch_size; ++i) {
    Dims tmp = input.tensor_shape(i);
    std::vector<int> input_shape(tmp.begin(), tmp.end());

    IntArr c_dims, c_perm;
    std::tie(c_dims, c_perm) = RowToColumnMajor(input_shape.data(),
                                                permutation.data(),
                                                input_shape.size());
    const void* in = input.raw_tensor(0);
    void* out = output.raw_mutable_tensor(0);
    cuttHandle plan;
    cuttCheck(cuttPlan(&plan, input_shape.size(), c_dims.get(), c_perm.get(), sizeof(T), stream));
    cuttCheck(cuttExecute(plan, in, out));
    CUDA_CALL(cudaStreamSynchronize(stream));
    cuttCheck(cuttDestroy(plan));
  }
}

/* We insert an additional dim iff batch_size > 1 because cuTT require dim0 to be > 1 */
template <typename T>
void cuTTKernelBatched(const TensorList<GPUBackend>& input,
                       TensorList<GPUBackend>& output,
                       const std::vector<int>& permutation,
                       cuttHandle* plan,
                       cudaStream_t stream) {
  int batch_size = static_cast<int>(input.ntensor());
  Dims tmp = input.tensor_shape(0);
  std::vector<int> input_shape(tmp.begin(), tmp.end());

  if (batch_size > 1) {
    input_shape.insert(input_shape.begin(), batch_size);
  }

  std::vector<int> batched_perm(permutation.begin(), permutation.end());
  if (batch_size > 1) {
    std::for_each(batched_perm.begin(), batched_perm.end(), [](int& i) { i++; });
    batched_perm.insert(batched_perm.begin(), 0);
  }

  IntArr c_dims, c_permutation;
  std::tie(c_dims, c_permutation) = RowToColumnMajor(input_shape.data(),
                                                     batched_perm.data(),
                                                     input_shape.size());

  if (*plan == 0) {
    cuttCheck(cuttPlan(plan,
                       batched_perm.size(),
                       c_dims.get(),
                       c_permutation.get(),
                       sizeof(T),
                       stream));
  }

  const void* in = input.raw_tensor(0);
  void* out = output.raw_mutable_tensor(0);
  cuttCheck(cuttExecute(*plan, in, out));
}
}  // namespace kernel

template <>
Transpose<GPUBackend>::~Transpose() {
  if (cutt_handle_ > 0) {
    cuttCheck(cuttDestroy(cutt_handle_));
  }
}

inline Dims GetPermutedDims(const Dims& dims, const std::vector<int>& permutation) {
  Dims permuted_dims;
  for (auto idx : permutation) {
    permuted_dims.push_back(dims[idx]);
  }
  return permuted_dims;
}

template<>
void Transpose<GPUBackend>::RunImpl(DeviceWorkspace* ws, int idx) {
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

  const auto& input = ws->Input<GPUBackend>(idx);
  auto& output = ws->Output<GPUBackend>(idx);

  DALI_ENFORCE((input.type().size() % 4) == 0,
      "cuTT transpose currently supports only 4-bytes aligned types.");

  output.set_type(input.type());
  // output.SetLayout(DALI_UNKNOWN);

  Dims input_shape = input.tensor_shape(0);
  DALI_ENFORCE(input_shape.size() == perm_.size(),
               "Transposed tensors rank should be equal to the permutation index list.");

  if (input.IsDenseTensor()) {
    if (cutt_handle_ != 0) {
      if (input_shape != previous_iter_shape_) {
        cuttCheck(cuttDestroy(cutt_handle_));
        cutt_handle_ = 0;
        previous_iter_shape_ = input_shape;
      }
    } else {
      previous_iter_shape_ = input_shape;
    }
    Dims permuted_dims = GetPermutedDims(input_shape, perm_);
    output.Resize(std::vector<Dims>(batch_size_, permuted_dims));
    if (input.type().size() == 4) {
      kernel::cuTTKernelBatched<int32_t>(input, output, perm_, &cutt_handle_, ws->stream());
    } else {  // == 8
      kernel::cuTTKernelBatched<int64_t>(input, output, perm_, &cutt_handle_, ws->stream());
    }
  } else {
    std::vector<Dims> tl_shape;
    for (int i = 0; i < batch_size_; ++i) {
      Dims in_shape = input.tensor_shape(i);
      tl_shape.emplace_back(GetPermutedDims(in_shape, perm_));
    }
    output.Resize(tl_shape);
    if (input.type().size() == 4) {
      kernel::cuTTKernel<int32_t>(input, output, perm_, ws->stream());
    } else {  // == 8
      kernel::cuTTKernel<int64_t>(input, output, perm_, ws->stream());
    }
  }

  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast< std::chrono::duration<double> >
                    (end - start).count();
  std::cout << "Transpose duration: " << (duration * 1000.0) << " ms" << std::endl;
}

DALI_REGISTER_OPERATOR(Transpose, Transpose<GPUBackend>, GPU);

}  // namespace dali
