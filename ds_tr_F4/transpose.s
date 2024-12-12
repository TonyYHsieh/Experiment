# TODO: modify
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Transpose
.globl Transpose
.p2align 8
.type Transpose,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Transpose
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 8 // accvgpr offset
  .amdhsa_next_free_vgpr 8 // vgprs
  .amdhsa_next_free_sgpr 24 // sgprs
  .amdhsa_group_segment_fixed_size 512 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text

.amdgpu_metadata
---
amdhsa.kernels:
- .agpr_count: 0
  .args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .offset: 16
    .size: 4
    .value_kind: by_value
  - .offset: 20
    .size: 4
    .value_kind: by_value
  .group_segment_fixed_size: 512
  .kernarg_segment_align: 8
  .kernarg_segment_size: 24
  .max_flat_workgroup_size: 256
  .name: Transpose
  .private_segment_fixed_size: 0
  .sgpr_count: 24
  .symbol: Transpose.kd
  .vgpr_count: 8
  .wavefront_size: 64
amdhsa.version:
- 1
- 1

.end_amdgpu_metadata

.set vgprSerial, 0
.set vgprOffset, 1
.set vgprValue,  2
.set vgprIdx0,   4
.set vgprIdx1,   5
.set vgprIdx2,   6
.set vgprTmp,    7

.set sgprKernelArg,   0
.set sgprWorkGroup0,  2
.set sgprWorkGroup1,  3
.set sgprWorkGroup2,  4
.set sgprSizeM,       5
.set sgprSizeK,       6
.set sgprAddressOut,  8
.set sgprAddressIn,   10
.set sgprSrdOut,      12
.set sgprSrdIn,       16
.set sgprTmp,         20

.set Srd127_96, 0x00020000

Transpose:
/* Load kernel args */
s_load_dwordx2 s[sgprAddressOut:sgprAddressOut+1],  s[sgprKernelArg:sgprKernelArg+1], 0
s_load_dwordx2 s[sgprAddressIn:sgprAddressIn+1],  s[sgprKernelArg:sgprKernelArg+1], 8
s_load_dword s[sgprSizeM], s[sgprKernelArg:sgprKernelArg+1], 16
s_load_dword s[sgprSizeK], s[sgprKernelArg:sgprKernelArg+1], 20
s_waitcnt lgkmcnt(0)

/* init_param */
s_mul_i32 s[sgprTmp], s[sgprSizeK], s[sgprSizeM]
s_lshr_b32 s[sgprTmp], s[sgprTmp], 1
s_mov_b32 s[sgprSrdOut+0], s[sgprAddressOut+0]
s_mov_b32 s[sgprSrdOut+1], s[sgprAddressOut+1]
s_mov_b32 s[sgprSrdOut+2], s[sgprTmp]
s_mov_b32 s[sgprSrdOut+3], Srd127_96

s_mul_i32 s[sgprTmp], s[sgprSizeM], s[sgprSizeK]
s_lshr_b32 s[sgprTmp], s[sgprTmp], 1
s_mov_b32 s[sgprSrdIn+0], s[sgprAddressIn+0]
s_mov_b32 s[sgprSrdIn+1], s[sgprAddressIn+1]
s_mov_b32 s[sgprSrdIn+2], s[sgprTmp]
s_mov_b32 s[sgprSrdIn+3], Srd127_96

v_mul_u32_u24 v[vgprOffset], v[vgprSerial], 16 // read 16 element
v_lshrrev_b32 v[vgprOffset], 1, v[vgprOffset]  // * 4 bits / 8 bits = / 2

buffer_load_dwordx2 v[vgprValue+0:vgprValue+1], v[vgprOffset], s[sgprSrdIn:sgprSrdIn+3], 0 offen offset:0
s_waitcnt vmcnt(0)

ds_write_b64 v[vgprOffset], v[vgprValue+0:vgprValue+1] offset:0

s_waitcnt lgkmcnt(0)

/* calculate ds read offset */
v_and_b32 v[vgprIdx0], v[vgprSerial], 0
v_and_b32 v[vgprIdx1], v[vgprSerial], 15
v_lshrrev_b32 v[vgprIdx1], 0, v[vgprIdx1]
v_lshrrev_b32 v[vgprIdx2], 4, v[vgprSerial]

v_mul_u32_u24 v[vgprIdx0], v[vgprIdx0], 16
v_mul_u32_u24 v[vgprIdx1], v[vgprIdx1], 16
v_mul_u32_u24 v[vgprIdx2], v[vgprIdx2], 16
v_mul_u32_u24 v[vgprIdx2], v[vgprIdx2], 16

v_add_u32 v[vgprOffset], v[vgprIdx0], v[vgprIdx1]
v_add_u32 v[vgprOffset], v[vgprOffset], v[vgprIdx2]
v_lshrrev_b32 v[vgprOffset], 1, v[vgprOffset]  // * 4 bits / 8 bits = / 2

ds_read_b64_tr_b4 v[vgprValue+0:vgprValue+1], v[vgprOffset] offset:0
# ds_read_b64 v[vgprValue+0:vgprValue+1], v[vgprOffset] offset:0

s_waitcnt lgkmcnt(0)

/* calculate buffer store offset */
v_and_b32 v[vgprIdx1], v[vgprSerial], 15
v_lshrrev_b32 v[vgprIdx0], 4, v[vgprSerial]

v_mul_u32_u24 v[vgprIdx0], v[vgprIdx0], 16
v_mul_u32_u24 v[vgprIdx1], v[vgprIdx1], s[sgprSizeK]// stride K 16 * 4
v_add_u32 v[vgprOffset], v[vgprIdx0], v[vgprIdx1]
v_lshrrev_b32 v[vgprOffset], 1, v[vgprOffset]  // * 4 bits / 8 bits = / 2

buffer_store_dwordx2 v[vgprValue+0:vgprValue+1], v[vgprOffset], s[sgprSrdOut:sgprSrdOut+3], 0 offen offset:0
s_waitcnt vmcnt(0)

s_endpgm
.LTranspose_end:
.size Transpose, .LTranspose_end - Transpose
