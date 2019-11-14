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

#ifndef DALI_OPERATORS_SEGMENTATION_POLY_TO_MASK_H_
#define DALI_OPERATORS_SEGMENTATION_POLY_TO_MASK_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"

namespace dali {

template <typename Backend>
class PolyToMask : public Operator<MixedWorkspace> {
 public:
  inline explicit PolyToMask(const OpSpec &spec)
    : Operator<MixedWorkspace>(spec) {

  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<MixedWorkspace> &ws) override;

  using dali::OperatorBase::Run;
  void Run(MixedWorkspace) override;

  bool CanInferOutputs() const override {
    return true;
  }

  kernels::KernelManager kmgr_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEGMENTATION_POLY_TO_MASK_H_
