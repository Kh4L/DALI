// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda.h>
#include "dali/error_handling.h"
#include "dali/util/cucontex.h"

namespace dali {

CUContext::CUContext() : context_{0}, initialized_{false} {
}

CUContext::CUContext(CUdevice device, unsigned int flags)
    : device_{device}, context_{0}, initialized_{false} {
    CUDA_CALL(cuInit(0));
    if (!CUDA_CALL(cuDevicePrimaryCtxRetain(&context_, device))) {
        throw std::runtime_error("cuDevicePrimaryCtxRetain failed, can't go forward without a context");
    }
    push();
    CUdevice dev;
    if (!CUDA_CALL(cuCtxGetDevice(&dev))) {
        throw std::runtime_error("Unable to get device");
    }
    initialized_ = true;
    CUDA_CALL(cuCtxSynchronize());
}

CUContext::CUContext(CUcontext ctx)
    : context_{ctx}, initialized_{true} {
}

CUContext::~CUContext() {
    if (initialized_) {
        // cuCtxPopCurrent?
        CUDA_CALL(cuDevicePrimaryCtxRelease(device_));
    }
}

CUContext::CUContext(CUContext&& other)
    : device_{other.device_}, context_{other.context_},
      initialized_{other.initialized_} {
    other.device_ = 0;
    other.context_ = 0;
    other.initialized_ = false;
}

CUContext& CUContext::operator=(CUContext&& other) {
    if (initialized_) {
        CUDA_CALL(cuCtxDestroy(context_));
    }
    device_ = other.device_;
    context_ = other.context_;
    initialized_ = other.initialized_;
    other.device_ = 0;
    other.context_ = 0;
    other.initialized_ = false;
    return *this;
}

void CUContext::push() const {
    CUcontext current;
    if (!CUDA_CALL(cuCtxGetCurrent(&current))) {
        throw std::runtime_error("Unable to get current context");
    }
    if (current != context_) {
        if (!CUDA_CALL(cuCtxPushCurrent(context_))) {
            throw std::runtime_error("Unable to push current context");
        }
    }
}

bool CUContext::initialized() const {
    return initialized_;
}

CUContext::operator CUcontext() const {
    return context_;
}

}  // namespace dali
