"""FlashJet as a Triton Inference Server Python-backend model.

A model directory that points at this class is written by
:mod:`flashjet.serving.model_repository`; see the serving guide for the whole
recipe.  The model contract is one event per request:

  inputs:   ``p4``   TYPE_FP64 [-1, 4]   (px, py, pz, E) per particle
            ``algo`` TYPE_FP64 [2]       (R, p)
  outputs:  ``jet_idx`` TYPE_INT32 [-1]  jet of each particle, beam-merge order
            ``n_jets``  TYPE_INT32 [1]

Every request Triton hands to :meth:`execute` that shares ``(R, p)`` is
clustered in one call, so enable ``dynamic_batching`` in the model
configuration: that batching is what makes the GPU kernels worth using.
"""

import json
import os

import numpy as np

try:  # only available inside the server
    import triton_python_backend_utils as pb_utils
except ImportError:  # pragma: no cover
    pb_utils = None

from . import cluster_events, resolve_backend


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        parameters = config.get("parameters", {})
        choice = parameters.get("backend", {}).get("string_value", "auto")
        self.backend = resolve_backend(choice)
        self.threads = int(os.environ.get("FLASHJET_THREADS", "1"))
        if pb_utils is not None:
            pb_utils.Logger.log_info(f"flashjet: backend={self.backend}, threads={self.threads}")

    def execute(self, requests):
        events = []
        for request in requests:
            p4 = pb_utils.get_input_tensor_by_name(request, "p4").as_numpy()
            algo = pb_utils.get_input_tensor_by_name(request, "algo").as_numpy().reshape(-1)
            events.append((p4.reshape(-1, 4), float(algo[0]), float(algo[1])))

        # one call per (R, p): the padding is shared, so are the kernels
        results = [None] * len(events)
        groups = {}
        for k, (_, R, p) in enumerate(events):
            groups.setdefault((R, p), []).append(k)
        for (R, p), members in groups.items():
            clustered = cluster_events(
                [events[k][0] for k in members], R=R, p=p, backend=self.backend, threads=self.threads
            )
            for k, result in zip(members, clustered):
                results[k] = result

        return [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("jet_idx", jet_idx.astype(np.int32).reshape(1, -1)),
                    pb_utils.Tensor("n_jets", np.array([[n_jets]], dtype=np.int32)),
                ]
            )
            for jet_idx, n_jets in results
        ]
