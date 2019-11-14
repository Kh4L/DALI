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

#include <map>
#include <vector>
#include "dali/operators/segmentation/poly_to_mask.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/masks.h"

namespace dali {

DALI_SCHEMA(PolyToMask)
.DocStr(R"code(Convert masks represented in COCO polygons format to binary pixels masks.
)code")
.NumInput(2)
.NumOutput(1);


using dali::OperatorBase::Run;
template <>
void PolyToMask<MixedBackend>::Run(MixedWorkspace &ws) {
  auto &output = ws.OutputRef<GPUBackend>(0);

  int h;
  int w;

  std::vector<int> sample_mask_meta_indices(batch_size_);
  std::vector<int> sample_poly_meta_indices(batch_size_);
  std::vector<int> sample_coords_indices(batch_size_);

  std::vector<int> mask_metas;
  std::vector<int> poly_metas;
  std::vector<float> coords;

  std::vector<std::vector<Index>> output_shape(batch_size_);

  for (int i = 0; i < batch_size_; ++i) {
    const auto& poly_meta = ws.InputRef<CPUBackend>(0, i);
    const auto& mask_coords = ws.InputRef<CPUBackend>(1, i);
    
    const int* poly_meta_data = poly_meta.data<int>()
    const float* mask_coords_data = mask_coords.data<float>();
    auto mask_indices_count = GetMaskIndicesCountFromMeta(poly_meta_data, poly_meta.size());
  
    int n_masks = mask_indices_count / 2;
    output_shape[i] = {n_masks, h, w};
  
    sample_mask_meta_indices.push_back(mask_metas.size());
    mask_metas.insert(mask_metas.end(), mask_indices_count.begin(), mask_indices_count.end());

    sample_poly_meta_indices.push_back(poly_metas.size());
    poly_metas.insert(poly_metas.end(), poly_meta_data, poly_meta_data + poly_meta.size());

    sample_coords_indices.push_back(poly_metas.size());
    coords.insert(coords.end(), mask_coords_data, mask_coords_data + mask_coords.size());
  }

  output.Resize(output_shape);

  // Copy on the GPU

  
/*
  DALI_TYPE_SWITCH_WITH_FP16(input.type().id(), DataType,
    VALUE_SWITCH(number_of_axes, NumAxes, (1, 2, 3, 4),
    (
      using Kernel = kernels::PolyToMaskGPU<DataType, NumAxes>;

      auto in_view = view<const DataType, NumAxes>(input);
      auto out_view = view<DataType, NumAxes>(output);
      kernels::KernelContext ctx;
      ctx.gpu.stream = ws.stream();
      kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view, static_cast<DataType>(fill_value_));
      // NOLINTNEXTLINE(whitespace/parens)
    ), DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_axes)););
  );  // NOLINT
*/
}

DALI_REGISTER_OPERATOR(PolyToMask, PolyToMask<MixedBackend>, Mixed);

}  // namespace dali
