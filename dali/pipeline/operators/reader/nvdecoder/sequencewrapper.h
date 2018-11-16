// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_
#define DALI_PIPELINE_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_

#include <condition_variable>
#include <mutex>
#include <thread>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/argument.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

// Struct that Loader::ReadOne will read
struct SequenceWrapper {
 public:

  explicit SequenceWrapper()
  : started_(false) {}

  void initialize(int count, int height, int width, int channels, cudaEvent_t event) {
    count = count;
    height = height;
    width = width;
    channels = channels;
    sequence.Resize({count, height, width, channels});
    started_ = true;
  }

  void set_started(cudaStream_t stream) {
    CUDA_CALL(cudaEventRecord(event_, stream));
    std::unique_lock<std::mutex> lock{started_lock_};
    started_ = true;
    lock.unlock();
    started_cv_.notify_one();
  }

  void wait() const {
    wait_until_started_();
    CUDA_CALL(cudaEventSynchronize(event_));
  }
  /*
  Remove useless?
  void wait(cudaStream_t stream) const {
      wait_until_started_();
      CUDA_CALL(cudaStreamWaitEvent(stream, event_, 0));
  }
  */
  Tensor<GPUBackend> sequence;
  int count;
  int height;
  int width;
  int channels;

 private:
  void wait_until_started_() const {
      std::unique_lock<std::mutex> lock{started_lock_};
      started_cv_.wait(lock, [&](){return started_;});
  }

  bool started_;
  mutable std::mutex started_lock_;
  mutable std::condition_variable started_cv_;
  cudaEvent_t event_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_