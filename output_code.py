# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/sv/csvuzarx4a3apcmql3jfxywryrluhzp7jpg2mhktf5asm5vmg7xq.py
# Topologically Sorted Source Nodes: [out, out_1, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out => convolution
#   out_1 => add_1, add_2, add_3, add_4, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, rsqrt, sub, var_mean
#   out_2 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.9), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.0012771392081736), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, 0.9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg5_1, %add_3), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_convolution_relu_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_convolution_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0', 'in_ptr3', 'in_ptr4', 'out_ptr3', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr5, xnumel, rnumel):
    xnumel = 6
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 784*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 784, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 784.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = 0.1
    tmp33 = tmp12 * tmp32
    tmp35 = 0.9
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = 1.0012771392081736
    tmp39 = tmp21 * tmp38
    tmp40 = tmp39 * tmp32
    tmp42 = tmp41 * tmp35
    tmp43 = tmp40 + tmp42
    tl.store(in_out_ptr0 + (r1 + 784*x0), tmp31, rmask)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr5 + (x0), tmp43, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xy/cxyd5jfbtvw7vfnenbgnbba7dq2hdu3ulj2tytnmjka4oit3bccq.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   out => convolution
#   out_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
#   out_2 => relu
#   out_3 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 14)
    x2 = xindex // 14
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1 + 56*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 56*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + 2*x1 + 56*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + 2*x1 + 56*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + 6*x3), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xq/cxqwlmbwtjlwkoy744vqyfdpqd7idt5uwzmrvs7bvhg6yvglzdgx.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   out => convolution
#   out_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
#   out_2 => relu
#   out_3 => _low_memory_max_pool2d_with_offsets
#   out_4 => convolution_1
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg8_1, %arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 6*x2 + 150*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lw/clwwb4ociz2fg5ln64vuaqaomlqshtcduz5bztujn4omzuwqzgsd.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4, out_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   out => convolution
#   out_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
#   out_2 => relu
#   out_3 => _low_memory_max_pool2d_with_offsets
#   out_4 => convolution_1
#   out_5 => add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, var_mean_1
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg8_1, %arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_1, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 0.1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg11_1, 0.9), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 1.0101010101010102), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 0.1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg12_1, 0.9), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg11_1, %add_7), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg12_1, %add_8), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr3', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 100
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r1), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 100, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.1
    tmp20 = tmp12 * tmp19
    tmp22 = 0.9
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = 100.0
    tmp26 = tmp18 / tmp25
    tmp27 = 1.0101010101010102
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28 * tmp19
    tmp31 = tmp30 * tmp22
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tl.store(out_ptr5 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tb/ctbykzxchd3cbqisapkpuuuglfr32kh2wxmvejh5olma46icnzuk.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4, out_5, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   out => convolution
#   out_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
#   out_2 => relu
#   out_3 => _low_memory_max_pool2d_with_offsets
#   out_4 => convolution_1
#   out_5 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
#   out_6 => relu_1
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg8_1, %arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_1, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %getitem_5), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_5), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_7), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 100.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lk/clk2sr4i6o2dh5vkniipbwv5q6xrxripy53rv5cb7bgd2v3pgiq7.py
# Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   out_9 => mul_tensor_1, sum_dim_int_list_1
# Graph fragment:
#   %mul_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_default_2, %unsqueeze_default_3), kwargs = {})
#   %sum_dim_int_list_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_tensor_1, [1]), kwargs = {})
triton_per_fused_addmm_5 = async_compile.triton('triton_per_fused_addmm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_addmm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_addmm_5(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 120
    XBLOCK: tl.constexpr = 1
    rnumel = 400
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((r1 % 5)) + 320*(((r1 // 5) % 5)) + (r1 // 25)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr0 + (16 + 32*((r1 % 5)) + 320*(((r1 // 5) % 5)) + (r1 // 25)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr0 + (160 + 32*((r1 % 5)) + 320*(((r1 // 5) % 5)) + (r1 // 25)), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr0 + (176 + 32*((r1 % 5)) + 320*(((r1 // 5) % 5)) + (r1 // 25)), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (r1 + 400*x0), rmask, other=0.0)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/th/cthmkljeywlcq3fkypcsloao73pwfk4ptbs3dxwrmqj35miguo3t.py
# Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   out_11 => add_tensor, mul_tensor, sum_dim_int_list
#   out_12 => relu_3
# Graph fragment:
#   %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_default, %unsqueeze_default_1), kwargs = {})
#   %sum_dim_int_list : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_tensor, [1]), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_dim_int_list, %arg18_1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
triton_per_fused_addmm_relu_6 = async_compile.triton('triton_per_fused_addmm_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_addmm_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_addmm_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 84
    rnumel = 120
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 120*x0), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp10 + tmp11
    tmp13 = triton_helpers.maximum(tmp3, tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/7k/c7k6nbhw5n2zs5gmaiaswgentsoaq6euufwydfancj76rh3rigpa.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg3_1, %add), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '9182018CCD6A4F758231D68D0B1E1E23CEBB32E5D78CB36B65791C4EB96774A2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1 = args
    args.clear()
    assert_size_stride(arg0_1, (6, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1_1, (6, ), (1, ))
    assert_size_stride(arg2_1, (1, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(arg3_1, (), ())
    assert_size_stride(arg4_1, (6, ), (1, ))
    assert_size_stride(arg5_1, (6, ), (1, ))
    assert_size_stride(arg6_1, (6, ), (1, ))
    assert_size_stride(arg7_1, (6, ), (1, ))
    assert_size_stride(arg8_1, (16, 6, 5, 5), (150, 25, 5, 1))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (), ())
    assert_size_stride(arg11_1, (16, ), (1, ))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (120, 400), (400, 1))
    assert_size_stride(arg16_1, (120, ), (1, ))
    assert_size_stride(arg17_1, (84, 120), (120, 1))
    assert_size_stride(arg18_1, (84, ), (1, ))
    assert_size_stride(arg19_1, (10, 84), (84, 1))
    assert_size_stride(arg20_1, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 6, 28, 28), (4704, 784, 28, 1))
        del arg0_1
        del arg2_1
        buf4 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_convolution_relu_0.run(buf4, arg1_1, arg6_1, arg7_1, arg4_1, arg5_1, arg4_1, arg5_1, 6, 784, grid=grid(6), stream=stream0)
        del arg1_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf5 = empty_strided_cuda((1, 6, 14, 14), (1176, 1, 84, 6), torch.float32)
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_1.run(buf4, buf5, 6, 196, grid=grid(6, 196), stream=stream0)
        del buf4
        buf6 = empty_strided_cuda((16, 6, 5, 5), (150, 1, 30, 6), torch.float32)
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_2.run(arg8_1, buf6, 96, 25, grid=grid(96, 25), stream=stream0)
        del arg8_1
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
        buf7 = extern_kernels.convolution(buf5, buf6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (1, 16, 10, 10), (1600, 1, 160, 16))
        del buf5
        del buf6
        buf8 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4, out_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_3.run(buf7, arg9_1, arg11_1, arg12_1, buf8, buf9, arg11_1, arg12_1, 16, 100, grid=grid(16), stream=stream0)
        del arg11_1
        del arg12_1
        buf11 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4, out_5, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_max_pool2d_with_indices_relu_4.run(buf11, arg9_1, buf8, buf9, arg13_1, arg14_1, 1600, grid=grid(1600), stream=stream0)
        del arg13_1
        del arg14_1
        del arg9_1
        del buf8
        del buf9
        buf12 = empty_strided_cuda((1, 120), (120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_per_fused_addmm_5.run(buf11, arg15_1, buf12, 120, 400, grid=grid(120), stream=stream0)
        del arg15_1
        del buf11
        buf13 = empty_strided_cuda((1, 84), (84, 1), torch.float32)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_addmm_relu_6.run(buf14, buf12, arg16_1, arg17_1, arg18_1, 84, 120, grid=grid(84), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        del buf12
        buf15 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg20_1, (1, 10), (0, 1), 0), buf14, reinterpret_tensor(arg19_1, (84, 10), (1, 84), 0), alpha=1, beta=1, out=buf15)
        del arg19_1
        del arg20_1
        del buf14
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_7.run(arg3_1, arg3_1, 1, grid=grid(1), stream=stream0)
        del arg3_1
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_7.run(arg10_1, arg10_1, 1, grid=grid(1), stream=stream0)
        del arg10_1
    return (buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((6, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, 6, 5, 5), (150, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg11_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((120, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((84, 120), (120, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((10, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
