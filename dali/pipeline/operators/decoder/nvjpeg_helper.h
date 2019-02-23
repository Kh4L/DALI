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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_

#include <nvjpeg.h>

#include <string>

#include "dali/common.h"
#include "dali/error_handling.h"

namespace dali {

#define NVJPEG_CALL(code)                                    \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      dali::string error = dali::string("NVJPEG error \"") + \
        std::to_string(static_cast<int>(status)) + "\"";     \
      DALI_FAIL(error);                                      \
    }                                                        \
  } while (0)

#define NVJPEG_CALL_EX(code, extra)                          \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    string extra_info = extra;                               \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      dali::string error = dali::string("NVJPEG error \"") + \
        std::to_string(static_cast<int>(status)) + "\"" +    \
        " " + extra_info;                                    \
      DALI_FAIL(error);                                      \
    }                                                        \
  } while (0)

  struct StateNvJPEG {
    nvjpegBackend_t nvjpeg_backend;
    nvjpegBufferPinned_t pinned_buffer;
    nvjpegDecoderStateHost_t decoder_host_state;
    nvjpegDecoderStateHost_t decoder_hybrid_state;
  };

  struct EncodedImageInfo {
    bool nvjpeg_support;
    int c;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
  };

  nvjpegOutputFormat_t GetFormat(DALIImageType type) {
    switch (type) {
      case DALI_RGB:
        return NVJPEG_OUTPUT_RGBI;
      case DALI_BGR:
        return NVJPEG_OUTPUT_BGRI;
      case DALI_GRAY:
        return NVJPEG_OUTPUT_Y;
      default:
        DALI_FAIL("Unknown output format");
    }
  }


  int GetOutputPitch(DALIImageType type) {
    switch (type) {
      case DALI_RGB:
      case DALI_BGR:
        return 3;
      case DALI_GRAY:
        return 1;
      default:
        DALI_FAIL("Unknown output format");
    }
  }

  bool SupportedSubsampling(const nvjpegChromaSubsampling_t &subsampling) {
    switch (subsampling) {
      case NVJPEG_CSS_444:
      case NVJPEG_CSS_422:
      case NVJPEG_CSS_420:
      case NVJPEG_CSS_411:
      case NVJPEG_CSS_410:
      case NVJPEG_CSS_GRAY:
      case NVJPEG_CSS_440:
        return true;
      default:
        return false;
    }
  }

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_HELPER_H_
