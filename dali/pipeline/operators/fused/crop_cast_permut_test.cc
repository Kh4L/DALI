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

#include "dali/test/dali_test_matching.h"

namespace dali {

template <typename ImgType>
class CropCastPermuteTest : public GenericMatchingTest<ImgType> {};


#define ENUM_TO_STRING(val) \
  std::to_string(static_cast<std::underlying_type<decltype(val)>::type>(val)).c_str()

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(CropCastPermuteTest, Types);

const bool addImageType = true;

TYPED_TEST(CropCastPermuteTest, CropVector) {
  this->RunTest({"CropCastPermute", {"crop", "224, 256", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(CropCastPermuteTest, Layout_DALI_NCHW) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_layout", ENUM_TO_STRING(DALI_NCHW), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Layout_DALI_NHWC) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_layout", ENUM_TO_STRING(DALI_NHWC), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Layout_DALI_SAME) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_layout", ENUM_TO_STRING(DALI_SAME), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_NO_TYPE) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_NO_TYPE), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_UINT8) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_UINT8), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_INT16) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_INT16), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_INT32) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_INT32), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_INT64) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_INT64), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_FLOAT16) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_FLOAT16), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_FLOAT) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", ENUM_TO_STRING(DALI_FLOAT), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

#undef LAYOUT_TO_STRING

}  // namespace dali
