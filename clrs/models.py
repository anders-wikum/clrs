# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""The CLRS Algorithmic Reasoning Benchmark."""

from clrs._src.baselines import BaselineModel
from clrs._src.baselines import BaselineModelChunked
from clrs._src.nets import Net
from clrs._src.nets import NetChunked
from clrs._src.processors import GAT
from clrs._src.processors import MPNN
from clrs._src.processors import MPNNDoubleMax

__all__ = (
    "BaselineModel",
    "BaselineModelChunked",
    "GAT",
    "MPNN",
    "MPNNDoubleMax", #TODO edited
    "Net",
    "NetChunked",
)
