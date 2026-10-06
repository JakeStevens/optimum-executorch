# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch
from executorch.exir.dialects._ops import ops

from optimum.exporters.executorch.recipes.xnnpack import _has_accumulating_index_put


@pytest.mark.parametrize(
    ("accumulate", "use_kwargs", "unsafe"),
    [(False, False, False), (True, False, True), (False, True, False), (True, True, True)],
)
def test_accumulating_index_put_guard(accumulate, use_kwargs, unsafe):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    indices = graph.placeholder("indices")
    values = graph.placeholder("values")
    args = (source, indices, values)
    kwargs = {"accumulate": accumulate} if use_kwargs else {}
    if not use_kwargs:
        args += (accumulate,)
    result = graph.call_function(ops.edge.aten.index_put.default, args=args, kwargs=kwargs)
    graph.output(result)

    assert _has_accumulating_index_put(SimpleNamespace(graph=graph)) is unsafe


def test_other_ops_do_not_disable_reinplace():
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    graph.output(source)

    assert not _has_accumulating_index_put(SimpleNamespace(graph=graph))
