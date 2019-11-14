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

#ifndef DALI_PIPELINE_UTIL_MASKS_H_
#define DALI_PIPELINE_UTIL_MASKS_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "dali/core/error_handling.h"

namespace dali {


/**
 * @brief Return a flat vector of pairs (start_idx, count) for a given sample.
 * The number of masks can then be obtained with `indices_vector.size() / 2`.
 */
std::vector<int> GetMaskIndicesCountFromMeta(const int* meta, int num_polygons) {
  std::vector<int> ret;
  int current_start_idx = 0;
  int current_mask_idx = meta[0];
  int i;
  for (i = 1; i < num_polygons; ++i) {
    if (meta[i * 3] != current_mask_idx) {
      ret.push_back(current_start_idx);
      ret.push_back(i - current_start_idx);
      current_start_idx = i;
      current_mask_idx = meta[i * 3];
    }
  }
  ret.push_back(current_start_idx);
  ret.push_back(i - current_start_idx);
  return ret;
}

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_MASKS_H_