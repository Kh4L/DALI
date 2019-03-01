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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_NEW_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_NEW_H_

#include <nvjpeg.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/decoder/nvjpeg_helper.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

class nvJPEGDecoderNew : public Operator<MixedBackend> {
 public:
  explicit nvJPEGDecoderNew(const OpSpec& spec) :
    Operator<MixedBackend>(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    output_info_(batch_size_),
    image_decoders_(batch_size_),
    image_states_(batch_size_),
    jpeg_streams_(batch_size_),
    decode_params_(batch_size_),
    pinned_buffer_(batch_size_),
    decoder_host_state_(batch_size_),
    decoder_huff_hybrid_state_(batch_size_),
    output_shape_(batch_size_),
    thread_pool_(num_threads_,
                 spec.GetArgument<int>("device_id"),
                 true /* pin threads */) {
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));

    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));

    // to create also in GPU Op
    NVJPEG_CALL(nvjpegDecoderCreate(
                      handle_, NVJPEG_BACKEND_HYBRID, &decoder_huff_host_));
    NVJPEG_CALL(nvjpegDecoderCreate(
                      handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_huff_hybrid_));


    for (int i = 0; i < batch_size_; i++) {
      NVJPEG_CALL(nvjpegDecodeParamsCreate(handle_, &decode_params_[i]));
      NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(decode_params_[i],
                                                    GetFormat(output_image_type_)));

      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &jpeg_streams_[i]));
      // We want to use nvJPEG default pinned allocator
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, nullptr, &pinned_buffer_[i]));

      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_huff_host_,
                                        &decoder_host_state_[i]));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_huff_hybrid_,
                                        &decoder_huff_hybrid_state_[i]));
    }

    // GPU
    // create the handles, streams and events we'll use
    // We want to use nvJPEG default device allocator
    NVJPEG_CALL(nvjpegBufferDeviceCreate(handle_, nullptr, &device_buffer_));
  }

  virtual ~nvJPEGDecoderNew() noexcept(false) {
    thread_pool_.WaitForWork();
    for (int i = 0; i < batch_size_; ++i) {
      NVJPEG_CALL(nvjpegJpegStreamDestroy(jpeg_streams_[i]));
      NVJPEG_CALL(nvjpegJpegStateDestroy(decoder_host_state_[i]));
      NVJPEG_CALL(nvjpegJpegStateDestroy(decoder_huff_hybrid_state_[i]));
    }
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_huff_host_));
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_huff_hybrid_));
    NVJPEG_CALL(nvjpegDestroy(handle_));
  }

  using dali::OperatorBase::Run;
  void Run(MixedWorkspace *ws) override {
    CPUStage(ws);
    GPUStage(ws);
  }

 protected:
  void CPUStage(MixedWorkspace *ws) {
    // Parsing and preparing metadata
    for (int i = 0; i < batch_size_; i++) {
      const auto &in = ws->Input<CPUBackend>(0, i);
      const auto *input_data = in.data<uint8_t>();
      const auto in_size = in.size();
      const auto file_name = in.GetSourceInfo();

      nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_,
                                                 static_cast<const unsigned char*>(input_data),
                                                 in_size,
                                                 false,
                                                 false,
                                                 jpeg_streams_[i]);

      EncodedImageInfo info;
      if (ret != NVJPEG_STATUS_SUCCESS) {
        try {
          const auto image = ImageFactory::CreateImage(
                                           static_cast<const uint8 *>(input_data), in_size);
          const auto dims = image->GetImageDims();
          info.heights[0] = std::get<0>(dims);
          info.widths[0] = std::get<1>(dims);
          info.nvjpeg_support = false;
          output_info_[i] = info;
        } catch (const std::runtime_error &e) {
          DALI_FAIL(e.what() + "File: " + file_name);
        }
      } else {
        NVJPEG_CALL(nvjpegJpegStreamGetFrameDimensions(jpeg_streams_[i],
                                                       info.widths,
                                                       info.heights));
        NVJPEG_CALL(nvjpegJpegStreamGetComponentsNum(jpeg_streams_[i],
                                                     &info.c));
        NVJPEG_CALL(nvjpegJpegStreamGetChromaSubsampling(jpeg_streams_[i],
                                                         &info.subsampling));
        info.nvjpeg_support = SupportedSubsampling(info.subsampling);
        if (info.nvjpeg_support) {
          if (ShouldBeHybrid(info)) {
            image_decoders_[i] = decoder_huff_hybrid_;
            image_states_[i] = decoder_huff_hybrid_state_[i];
          } else {
            image_decoders_[i] = decoder_huff_host_;
            image_states_[i] = decoder_host_state_[i];
          }

          /*
          // TODO(spanev): add this function when integrating Crop
          nvjpegnvjpegDecodeParamsSetROI(decode_params_[pos], offset_x, offset_y, roi_w, roi_h);
          */
        }
      }
      const auto c = static_cast<Index>(NumberOfChannels(output_image_type_));
      output_shape_[i] = Dims({info.heights[0], info.widths[0], c});
      output_info_[i] = info;
    }

    // Creating output shape and setting the order of images so the largest are processed first
    // (for load balancing)
    std::vector<std::pair<size_t, size_t>> image_order(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      image_order[i] = std::make_pair(volume(output_shape_[i]), i);
    }
    std::sort(image_order.begin(), image_order.end(),
              std::greater<std::pair<size_t, size_t>>());
    // Running the CPU stage
    for (int idx = 0; idx < batch_size_; ++idx) {
      const int i = image_order[idx].second;
      EncodedImageInfo& info = output_info_[i];
      if (info.nvjpeg_support) {
        const auto &in = ws->Input<CPUBackend>(0, i);
        const auto file_name = in.GetSourceInfo();
        thread_pool_.DoWorkWithID(std::bind(
              [this, i, file_name](int idx, int tid) {
                CPUSampleWorker(i, file_name);
              }, i, std::placeholders::_1));
      }
    }

    thread_pool_.WaitForWork();
  }

  void CPUSampleWorker(int i, string file_name) {
    EncodedImageInfo& info = output_info_[i];
    NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(image_states_[i],
                                              pinned_buffer_[i]));
    nvjpegStatus_t ret = nvjpegDecodeJpegHost(handle_,
                                              image_decoders_[i],
                                              image_states_[i],
                                              decode_params_[i],
                                              jpeg_streams_[i]);
    // If image is somehow not supported try hostdecoder
    if (ret != NVJPEG_STATUS_SUCCESS) {
      if (ret == NVJPEG_STATUS_JPEG_NOT_SUPPORTED || ret == NVJPEG_STATUS_BAD_JPEG) {
        info.nvjpeg_support = false;
      } else {
        NVJPEG_CALL_EX(ret, file_name);
      }
    }
  }

  void GPUStage(MixedWorkspace* ws) {
    auto& output = ws->Output<GPUBackend>(0);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output.set_type(type);
    output.Resize(output_shape_);
    output.SetLayout(DALI_NHWC);

    for (int i = 0; i < batch_size_; i++) {
      const EncodedImageInfo& info = output_info_[i];
      auto *output_data = output.mutable_tensor<uint8_t>(i);

      if (info.nvjpeg_support) {
        nvjpegImage_t nvjpeg_image;
        nvjpeg_image.channel[0] = output_data;
        nvjpeg_image.pitch[0] = NumberOfChannels(output_image_type_) * info.widths[0];

        NVJPEG_CALL(nvjpegStateAttachDeviceBuffer(image_states_[i], device_buffer_));

        NVJPEG_CALL(nvjpegDecodeJpegTransferToDevice(
            handle_,
            image_decoders_[i],
            image_states_[i],
            jpeg_streams_[i],
            ws->stream()));

        NVJPEG_CALL(nvjpegDecodeJpegDevice(
            handle_,
            image_decoders_[i],
            image_states_[i],
            &nvjpeg_image,
            ws->stream()));
      } else {
        auto& in = ws->Input<CPUBackend>(0, i);
        const auto* input_data = in.data<uint8_t>();
        const auto& file_name = in.GetSourceInfo();
        OCVFallback(input_data, in.size(), output_data, ws->stream(), file_name);
      }
    }
  }

  bool ShouldBeHybrid(EncodedImageInfo& info) {
    return info.widths[0] * info.heights[0] > hybrid_huffman_threshold_;
  }
  /**
   // Running the CPU stage
   * Fallback to openCV's cv::imdecode for all images nvjpeg can't handle
   */
  void OCVFallback(const uint8_t *data, int size,
                   uint8_t *decoded_device_data, cudaStream_t s, string file_name) {
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

    CUDA_CALL(cudaMemcpyAsync(decoded_device_data,
                              tmp.ptr(),
                              tmp.rows * tmp.cols * c,
                              cudaMemcpyHostToDevice, s));
  }

  USE_OPERATOR_MEMBERS();
  nvjpegHandle_t handle_;

  // output colour format
  DALIImageType output_image_type_;

  unsigned hybrid_huffman_threshold_;

  // Common
  // Storage for per-image info
  std::vector<EncodedImageInfo> output_info_;
  nvjpegJpegDecoder_t decoder_huff_host_;
  nvjpegJpegDecoder_t decoder_huff_hybrid_;

  // CPU
  std::vector<nvjpegJpegDecoder_t> image_decoders_;
  std::vector<nvjpegJpegState_t> image_states_;
  std::vector<nvjpegJpegStream_t> jpeg_streams_;
  std::vector<nvjpegDecodeParams_t> decode_params_;
  std::vector<nvjpegBufferPinned_t> pinned_buffer_;
  std::vector<nvjpegJpegState_t> decoder_host_state_;
  std::vector<nvjpegJpegState_t> decoder_huff_hybrid_state_;
  std::vector<Dims> output_shape_;

  // GPU
  nvjpegBufferDevice_t device_buffer_;

  ThreadPool thread_pool_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_NEW_H_
