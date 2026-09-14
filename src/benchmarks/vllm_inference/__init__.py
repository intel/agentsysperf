#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""vLLM CPU inference benchmark adapter for AgentSysPerf.

Drives vLLM serving benchmarks with separate prefill and decode
phase measurement, enabling the MemoryBandwidthAnalyzer to identify
speculative decoding opportunities.

Tuning reference: vLLM's CPU installation and performance guidance,
https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
"""

from .adapter import VLLMInferenceAdapter

__all__ = ["VLLMInferenceAdapter"]
