# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import os
import glob

_tf_plugins = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libdali_tf*.so'))
_dali_tf_module = None
for _libdali_tf in _tf_plugins:
  try:
    _dali_tf_module = tf.load_op_library(_libdali_tf)
    break
  # if plugin is not compatible skip it
  except tf.errors.NotFoundError:
    pass
else:
  raise Exception('No matching DALI plugin found for installed TensorFlow version')

_dali_tf = _dali_tf_module.dali

def DALIIteratorWrapper(pipeline, **kwargs):
  serialized_pipeline = pipeline.serialize()
  del pipeline
  return _dali_tf(serialized_pipeline=serialized_pipeline, **kwargs)


def DALIIterator():
    return DALIIteratorWrapper

def DALISerializedIterator():
    return _dali_tf

op_doc = _dali_tf.__doc__
wrapper_doc = op_doc.replace('serialized DALI pipeline (given in `serialized_pipeline` parameter)', 'DALI pipeline').replace('serialized_pipeline: A `string`', 'pipeline: A `Pipeline` object')

DALIIterator.__doc__ = wrapper_doc
DALIIteratorWrapper.__doc__ = wrapper_doc
DALISerializedIterator.__doc__ = _dali_tf.__doc__
