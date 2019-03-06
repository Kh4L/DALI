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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_CPU_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_CPU_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/decoder/nvjpeg_helper.h"

#include "dali/image/image_factory.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"

namespace dali {

class nvJPEGDecoderCPUStage : public Operator<CPUBackend> {
 public:
  explicit nvJPEGDecoderCPUStage(const OpSpec& spec) :
    Operator<CPUBackend>(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    decode_params_(batch_size_) {
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));

    // Do we really need both in both stages ops?
    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));

    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_HYBRID, &decoder_host_));
    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_hybrid_));

    for (int i = 0; i < batch_size_; i++) {
      NVJPEG_CALL(nvjpegDecodeParamsCreate(handle_, &decode_params_[i]));
      NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(decode_params_[i],
                                                    GetFormat(output_image_type_)));
      NVJPEG_CALL(nvjpegDecodeParamsSetAllowCMYK(decode_params_[i], true));
    }
  }

  virtual ~nvJPEGDecoderCPUStage() noexcept(false) {
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_host_));
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_hybrid_));
    NVJPEG_CALL(nvjpegDestroy(handle_));
  }

  void RunImpl(SampleWorkspace *ws, const int idx) {
    const int data_idx = ws->data_idx();
    const auto& in = ws->Input<CPUBackend>(0);
    const auto *input_data = in.data<uint8_t>();
    const auto in_size = in.size();
    const auto file_name = in.GetSourceInfo();

    // We init and store the images and nvJPEG state in the CPU operator outputs
    // Since this CPU operator is instatiated on the GPU/Mixed boundary, it allows us to use
    // the Executor CPU output multiple-buffering

    // Output #0 contains general infomation about the sample
    // Output #1 contains nvJPEG state, buffer and information
    // Output #3 optionally contains OpenCV if nvJPEG cannot decode the sample

    EncodedImageInfo* info;
    StateNvJPEG* state_nvjpeg;
    std::tie(info, state_nvjpeg) = InitAndGet(ws->Output<CPUBackend>(0),
                                              ws->Output<CPUBackend>(1));

    nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_,
                                                static_cast<const unsigned char*>(input_data),
                                                in_size,
                                                false,
                                                false,
                                                state_nvjpeg->jpeg_stream);
    info->nvjpeg_support = ret == NVJPEG_STATUS_SUCCESS;
    if (!info->nvjpeg_support) {
      try {
        const auto image = ImageFactory::CreateImage(static_cast<const uint8 *>(input_data),
                                                     in_size);
        const auto dims = image->GetImageDims();
        info->heights[0] = std::get<0>(dims);
        info->widths[0] = std::get<1>(dims);
        auto& out = ws->Output<CPUBackend>(2);
        out.set_type(TypeInfo::Create<uint8_t>());
        const auto c = static_cast<Index>(NumberOfChannels(output_image_type_));
        out.Resize({info->heights[0], info->widths[0], c});
        auto *output_data = out.mutable_data<uint8_t>();

        OCVFallback(input_data, in_size, output_data, file_name);
      } catch (const std::runtime_error &e) {
        DALI_FAIL(e.what() + "File: " + file_name);
      }
    } else {
      NVJPEG_CALL(nvjpegJpegStreamGetFrameDimensions(state_nvjpeg->jpeg_stream,
                                                     info->widths,
                                                     info->heights));
      NVJPEG_CALL(nvjpegJpegStreamGetComponentsNum(state_nvjpeg->jpeg_stream,
                                                   &info->c));
      state_nvjpeg->nvjpeg_backend =
                    ShouldBeHybrid(*info) ? NVJPEG_BACKEND_GPU_HYBRID : NVJPEG_BACKEND_HYBRID;

      /* TODO(spanev): add this function when integrating Crop
      nvjpegnvjpegDecodeParamsSetROI(decode_params_[i], offset_x, offset_y, roi_w, roi_h); */

      nvjpegJpegState_t state = GetNvjpegState(*state_nvjpeg);
      NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(state,
                                                state_nvjpeg->pinned_buffer));
      nvjpegStatus_t ret = nvjpegDecodeJpegHost(
          handle_,
          GetDecoder(state_nvjpeg->nvjpeg_backend),
          state,
          decode_params_[data_idx],
          state_nvjpeg->jpeg_stream);
      if (ret != NVJPEG_STATUS_SUCCESS) {
        if (ret == NVJPEG_STATUS_JPEG_NOT_SUPPORTED || ret == NVJPEG_STATUS_BAD_JPEG) {
          info->nvjpeg_support = false;
        } else {
          NVJPEG_CALL_EX(ret, file_name);
        }
      } else {
        // TODO(spanev): free Output #2
      }
    }
  }

 protected:
  USE_OPERATOR_MEMBERS();

  inline std::pair<EncodedImageInfo*, StateNvJPEG*>
  InitAndGet(Tensor<CPUBackend>& info_tensor, Tensor<CPUBackend>& state_tensor) {
    if (info_tensor.size() == 0) {
      TypeInfo type;
      // we need to set a arbitrary to be able to access to call raw_data()
      type.SetType<uint8_t>();

      std::shared_ptr<EncodedImageInfo> info_p(new EncodedImageInfo());
      info_tensor.ShareData(info_p, 1, {1});
      info_tensor.set_type(type);

      std::shared_ptr<StateNvJPEG> state_p(new StateNvJPEG(),
        [](StateNvJPEG* s) {
          NVJPEG_CALL(nvjpegJpegStreamDestroy(s->jpeg_stream));
          NVJPEG_CALL(nvjpegBufferPinnedDestroy(s->pinned_buffer));
          NVJPEG_CALL(nvjpegJpegStateDestroy(s->decoder_host_state));
          NVJPEG_CALL(nvjpegJpegStateDestroy(s->decoder_hybrid_state));
      });

      // We want to use nvJPEG default pinned allocator
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, nullptr, &state_p->pinned_buffer));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_host_,
                                        &state_p->decoder_host_state));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_hybrid_,
                                        &state_p->decoder_hybrid_state));
      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &state_p->jpeg_stream));

      state_tensor.ShareData(state_p, 1, {1});
      state_tensor.set_type(type);
    }

    EncodedImageInfo* info = reinterpret_cast<EncodedImageInfo*>(info_tensor.raw_mutable_data());
    StateNvJPEG* nvjpeg_state = reinterpret_cast<StateNvJPEG*>(state_tensor.raw_mutable_data());
    return std::make_pair(info, nvjpeg_state);
  }

  bool ShouldBeHybrid(EncodedImageInfo& info) const {
    return info.widths[0] * info.heights[0] > hybrid_huffman_threshold_;
  }

  inline nvjpegJpegDecoder_t GetDecoder(nvjpegBackend_t backend) const {
    switch (backend) {
      case NVJPEG_BACKEND_HYBRID:
        return decoder_host_;
      case NVJPEG_BACKEND_GPU_HYBRID:
        return decoder_hybrid_;
      default:
        DALI_FAIL("Unknown nvjpegBackend_t " + std::to_string(backend));
        return nullptr;
    }
  }

  /**
   * Fallback to openCV's cv::imdecode for all images nvjpeg can't handle
   */
  void OCVFallback(const uint8_t* data, int size,
                   uint8_t *decoded_host_data, string file_name) {
    const int c = (output_image_type_ == DALI_GRAY) ? 1 : 3;
    auto decode_type = (output_image_type_ == DALI_GRAY) ? cv::IMREAD_GRAYSCALE
                                                         : cv::IMREAD_COLOR;
    cv::Mat input(1,
                  size,
                  CV_8UC1,
                  reinterpret_cast<unsigned char*>(const_cast<uint8_t*>(data)));
    cv::Mat tmp = cv::imdecode(input, decode_type);

    if (tmp.data == nullptr) {
      DALI_FAIL("Unsupported image type: " + file_name);
    }

    // Transpose BGR -> output_image_type_ if needed
    if (IsColor(output_image_type_) && output_image_type_ != DALI_BGR) {
      OpenCvColorConversion(DALI_BGR, tmp, output_image_type_, tmp);
    }

    std::memcpy(decoded_host_data,
                tmp.ptr(),
                tmp.rows * tmp.cols * c);
  }

  // output colour format
  DALIImageType output_image_type_;

  unsigned hybrid_huffman_threshold_;

  // Common handles
  nvjpegHandle_t handle_;
  nvjpegJpegDecoder_t decoder_host_;
  nvjpegJpegDecoder_t decoder_hybrid_;

  // TODO(spanev): add huffman hybrid decode
  std::vector<nvjpegDecodeParams_t> decode_params_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_CPU_H_
