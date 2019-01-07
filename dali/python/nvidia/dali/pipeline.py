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

#pylint: disable=no-member
from collections import deque
from nvidia.dali import backend as b
from nvidia.dali import edge

class Pipeline(object):
    """Pipeline class encapsulates all data required to define and run
    DALI input pipeline.

    Parameters
    ----------
    `batch_size` : int, optional, default = -1
                   Batch size of the pipeline. Negative values for this parameter
                   are invalid - the default value may only be used with
                   serialized pipeline (the value stored in serialized pipeline
                   is used instead).
    `num_threads` : int, optional, default = -1
                    Number of CPU threads used by the pipeline.
                    Negative values for this parameter are invalid - the default
                    value may only be used with serialized pipeline (the value
                    stored in serialized pipeline is used instead).
    `device_id` : int, optional, default = -1
                  Id of GPU used by the pipeline.
                  Negative values for this parameter are invalid - the default
                  value may only be used with serialized pipeline (the value
                  stored in serialized pipeline is used instead).
    `seed` : int, optional, default = -1
             Seed used for random number generation. Leaving the default value
             for this parameter results in random seed.
    `exec_pipelined` : bool, optional, default = True
                       Whether to execute the pipeline in a way that enables
                       overlapping CPU and GPU computation, typically resulting
                       in faster execution speed, but larger memory consumption.
    `exec_async` : bool, optional, default = True
                   Whether to execute the pipeline asynchronously.
                   This makes :meth:`nvidia.dali.pipeline.Pipeline.run` method
                   run asynchronously with respect to the calling Python thread.
                   In order to synchronize with the pipeline one needs to call
                   :meth:`nvidia.dali.pipeline.Pipeline.outputs` method.
    `bytes_per_sample` : int, optional, default = 0
                         A hint for DALI for how much memory to use for its tensors.
    `set_affinity` : bool, optional, default = False
                     Whether to set CPU core affinity to the one closest to the
                     GPU being used.
    `max_streams` : int, optional, default = -1
                    Limit the number of CUDA streams used by the executor.
                    Value of -1 does not impose a limit.
                    This parameter is currently unused (and behavior of
                    unrestricted number of streams is assumed).
    `prefetch_queue_depth`: int, optional, default = 2
                            Depth of the executor pipeline. Deeper pipeline makes DALI
                            more resistant to uneven execution time of each batch, but it
                            also consumes more memory for internal buffers.
    """
    def __init__(self, batch_size = -1, num_threads = -1, device_id = -1, seed = -1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 set_affinity=False, max_streams=-1):
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._prefetch_queue_depth = prefetch_queue_depth
        self._built = False
        self._first_iter = True
        self._prepared = False
        self._names_and_devices = None
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._set_affinity = set_affinity
        self._max_streams = max_streams

    @property
    def batch_size(self):
        """Batch size."""
        return self._batch_size

    @property
    def num_threads(self):
        """Number of CPU threads used by the pipeline."""
        return self._num_threads

    @property
    def device_id(self):
        """Id of the GPU used by the pipeline."""
        return self._device_id

    def epoch_size(self, name = None):
        """Epoch size of a pipeline.

        If the `name` parameter is `None`, returns a dictionary of pairs
        `(reader name, epoch size for that reader)`.
        If the `name` parameter is not `None`, returns epoch size for that
        reader.

        Parameters
        ----------
        name : str, optional, default = None
               The reader which should be used to obtain epoch size.
        """

        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if name is not None:
            return self._pipe.epoch_size(name)
        return self._pipe.epoch_size()

    def _prepare_graph(self):
        self._pipe = b.Pipeline(self._batch_size,
                                self._num_threads,
                                self._device_id,
                                self._seed,
                                self._exec_pipelined,
                                self._prefetch_queue_depth,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams)
        outputs = self.define_graph()
        if (not isinstance(outputs, tuple) and
            not isinstance(outputs, list)):
            outputs = (outputs,)

        for output in outputs:
            if not isinstance(output, edge.EdgeReference):
                raise TypeError(
                    ("Expected outputs of type "
                    "EdgeReference. Received "
                    "output type {}")
                    .format(type(output).__name__)
                )

        # Backtrack to construct the graph
        op_ids = set()
        edges = deque(outputs)
        ops = []
        while edges:
            current_edge = edges.popleft()
            source_op = current_edge.source
            if source_op is None:
                raise RuntimeError(
                    "Pipeline encountered "
                    "Edge with no source op.")

            # To make sure we don't double count ops in
            # the case that they produce more than one
            # output, we keep track of the unique op ids
            # for each op we encounter and only add the
            # op if we have not already
            if source_op.id not in op_ids:
                op_ids.add(source_op.id)
                source_op.check_args()
                ops.append(source_op)
            else:
                # If the op was already added, we need to
                # change its position to the top of the list.
                # This ensures topological ordering of ops
                # when adding to the backend pipeline
                ops.remove(source_op)
                ops.append(source_op)
            for edge in source_op.inputs:
                if isinstance(edge, list):
                    for edge in edge:
                        edges.append(e)
                else:
                    edges.append(edge)

        # Add the ops to the graph and build the backend
        while ops:
            op = ops.pop()
            self._pipe.AddOperator(op.spec, op.name)
        self._prepared = True
        self._names_and_devices = [(e.name, e.device) for e in outputs]

    def build(self):
        """Build the pipeline.

        Pipeline needs to be built in order to run it standalone.
        Framework-specific plugins handle this step automatically.
        """
        if self._built:
            return

        if not self._prepared:
            self._prepare_graph()

        self._pipe.Build(self._names_and_devices)
        self._built = True

    def feed_input(self, ref, data):
        """Bind the NumPy array to a tensor produced by ExternalSource
        operator."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if not isinstance(ref, edge.EdgeReference):
            raise TypeError(
                ("Expected argument one to "
                "be EdgeReference. "
                "Received output type {}")
                .format(type(ref).__name__)
            )
        if isinstance(data, list):
            inputs = []
            for datum in data:
                inputs.append(edge.TensorCPU(datum))
            self._pipe.SetExternalTensorInput(ref.name, inputs)
        else:
            inp = edge.TensorListCPU(data)
            self._pipe.SetExternalTLInput(ref.name, inp)

    def _run_cpu(self):
        """Run CPU portion of the pipeline."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.RunCPU()

    def _run_gpu(self):
        """Run GPU portion of the pipeline."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.RunGPU()

    def outputs(self):
        """Returns the outputs of the pipeline and releases previous buffer.

        If the pipeline is executed asynchronously, this function blocks
        until the results become available."""
        self._release_outputs()
        return self._share_outputs()

    def _share_outputs(self):
        """Returns the outputs of the pipeline.

        Main difference to outputs is that _share_outputs doesn't release
        returned buffers, _release_outputs need to be called for that.
        If the pipeline is executed asynchronously, this function blocks
        until the results become available."""
        return self._pipe.ShareOutputs()

    def _release_outputs(self):
        """Release buffers returned by _share_outputs calls.

        It helps in case when output call result is consumed (copied)
        and buffers can be set free before next _share_outputs call"""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        return self._pipe.ReleaseOutputs()

    def run(self):
        """Run the pipeline and return the result.

        If the pipeline was created with `exec_async` option set to `True`,
        this function will also start prefetching the next iteration for
        faster execution."""
        if self._first_iter and self._exec_pipelined:
            self._prefetch()
        else:
            self._start_run()
        return self.outputs()

    def _prefetch(self):
        """Executes pipeline to fill executor's pipeline."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if self._first_iter and self._exec_pipelined:
            self._first_iter = False
            for i in range(self._prefetch_queue_depth):
                self._start_run()

    def _start_run(self):
        """Start running the pipeline without waiting for its results.

        If the pipeline was created with `exec_async` option set to `True`,
        this function will return without waiting for the execution to end."""
        self.iter_setup()
        self._run_cpu()
        self._run_gpu()

    def serialize(self):
        """Serialize the pipeline to a Protobuf string."""
        if not self._prepared:
            self._prepare_graph()
            self._pipe.SetOutputNames(self._names_and_devices)
        return self._pipe.SerializeToProtobuf()

    def deserialize_and_build(self, serialized_pipeline):
        """Deserialize and build the pipeline given in serialized form.

        Parameters
        ----------
        serialized_pipeline : str
                              Serialized pipeline.
        """
        self._pipe = b.Pipeline(serialized_pipeline,
                                self._batch_size,
                                self._num_threads,
                                self._device_id,
                                self._exec_pipelined,
                                self._prefetch_queue_depth,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams)
        self._prepared = True
        self._pipe.Build()
        self._built = True

    def save_graph_to_dot_file(self, filename):
        """Saves the pipeline graph to a file.

        Parameters
        ----------
        filename : str
                   Name of the file to which the graph is written.
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.SaveGraphToDotFile(filename)

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.

        It returns a list of output `EdgeReference`."""
        raise NotImplementedError

    def iter_setup(self):
        """This function can be overriden by user-defined
        pipeline to perform any needed setup for each iteration.
        For example, one can use this function to feed the input
        data from NumPy arrays."""
        pass
