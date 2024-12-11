# TODO: modify
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected MatMul
.globl MatMul
.p2align 8
.type MatMul,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel MatMul
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 32 // accvgpr offset
  .amdhsa_next_free_vgpr 40 // vgprs
  .amdhsa_next_free_sgpr 48 // sgprs
  .amdhsa_group_segment_fixed_size 32 // lds bytes
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
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 24
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 32
    .size: 8
    .value_kind: global_buffer
  - .offset: 40
    .size: 4
    .value_kind: by_value
  - .offset: 44
    .size: 4
    .value_kind: by_value
  - .offset: 48
    .size: 4
    .value_kind: by_value
  .group_segment_fixed_size: 16
  .kernarg_segment_align: 8
  .kernarg_segment_size: 56
  .max_flat_workgroup_size: 256
  .name: MatMul
  .private_segment_fixed_size: 0
  .sgpr_count: 48
  .symbol: MatMul.kd
  .vgpr_count: 40
  .wavefront_size: 64
amdhsa.version:
- 1
- 1

.end_amdgpu_metadata

.set vgprSerial,  0
.set vgprOffset,  1
.set vgprIndexT,  2  // tile
.set vgprIndexU,  3  // unroll
.set vgprValueA,  4
.set vgprValueB,  12
.set vgprValueC,  20
.set vgprValueSA, 24
.set vgprValueSB, 25
.set vgprTmp,     26

.set sgprKernelArg,   0
.set sgprWorkGroup0,  2
.set sgprWorkGroup1,  3
.set sgprWorkGroup2,  4
.set sgprSizeM,       5
.set sgprSizeN,       6
.set sgprSizeK,       7
.set sgprAddressC,    8
.set sgprAddressA,   10
.set sgprAddressSA,  12
.set sgprAddressB,   14
.set sgprAddressSB,  16

.set sgprSrcA,       20
.set sgprSrcSA,      24
.set sgprSrcB,       28
.set sgprSrcSB,      32
.set sgprDstC,       36
.set sgprTmp,        40

.set Srd127_96, 0x00020000


MatMul:
/* Load kernel args */
s_load_dwordx2 s[sgprAddressC:sgprAddressC+1],  s[sgprKernelArg:sgprKernelArg+1], 0
s_load_dwordx2 s[sgprAddressA:sgprAddressA+1],  s[sgprKernelArg:sgprKernelArg+1], 8
s_load_dwordx2 s[sgprAddressSA:sgprAddressSA+1], s[sgprKernelArg:sgprKernelArg+1], 16
s_load_dwordx2 s[sgprAddressB:sgprAddressB+1],  s[sgprKernelArg:sgprKernelArg+1], 24
s_load_dwordx2 s[sgprAddressSB:sgprAddressSB+1], s[sgprKernelArg:sgprKernelArg+1], 32
s_load_dword s[sgprSizeM], s[sgprKernelArg:sgprKernelArg+1], 40
s_load_dword s[sgprSizeN], s[sgprKernelArg:sgprKernelArg+1], 44
s_load_dword s[sgprSizeK], s[sgprKernelArg:sgprKernelArg+1], 48
s_waitcnt lgkmcnt(0)


# s_mul_i32 s[sgprTmp], s[sgprWorkGroup0], s[sgprTmp]

/* init_param */
s_mul_i32 s[sgprTmp], s[sgprSizeK], s[sgprSizeM]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 2  // * 4 bits per elements
s_lshr_b32 s[sgprTmp], s[sgprTmp], 3  // / 8 bits per bytes
s_mov_b32 s[sgprSrcA+0], s[sgprAddressA+0]
s_mov_b32 s[sgprSrcA+1], s[sgprAddressA+1]
s_mov_b32 s[sgprSrcA+2], s[sgprTmp]
s_mov_b32 s[sgprSrcA+3], Srd127_96

s_mul_i32 s[sgprTmp], s[sgprSizeK], s[sgprSizeM]
s_lshr_b32 s[sgprTmp], s[sgprTmp], 5
s_mov_b32 s[sgprSrcSA+0], s[sgprAddressSA+0]
s_mov_b32 s[sgprSrcSA+1], s[sgprAddressSA+1]
s_mov_b32 s[sgprSrcSA+2], s[sgprTmp]
s_mov_b32 s[sgprSrcSA+3], Srd127_96

s_mul_i32 s[sgprTmp], s[sgprSizeK], s[sgprSizeN]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 2  // * 4 bits per elements
s_lshr_b32 s[sgprTmp], s[sgprTmp], 3  // / 8 bits per bytes
s_mov_b32 s[sgprSrcB+0], s[sgprAddressB+0]
s_mov_b32 s[sgprSrcB+1], s[sgprAddressB+1]
s_mov_b32 s[sgprSrcB+2], s[sgprTmp]
s_mov_b32 s[sgprSrcB+3], Srd127_96

s_mul_i32 s[sgprTmp], s[sgprSizeK], s[sgprSizeN]
s_lshr_b32 s[sgprTmp], s[sgprTmp], 5
s_mov_b32 s[sgprSrcSB+0], s[sgprAddressSB+0]
s_mov_b32 s[sgprSrcSB+1], s[sgprAddressSB+1]
s_mov_b32 s[sgprSrcSB+2], s[sgprTmp]
s_mov_b32 s[sgprSrcSB+3], Srd127_96

s_mul_i32 s[sgprTmp], s[sgprSizeM], s[sgprSizeN]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 2
s_mov_b32 s[sgprDstC+0], s[sgprAddressC+0]
s_mov_b32 s[sgprDstC+1], s[sgprAddressC+1]
s_mov_b32 s[sgprDstC+2], s[sgprTmp]
s_mov_b32 s[sgprDstC+3], Srd127_96

v_and_b32 v[vgprIndexT], v[vgprSerial], 15
v_lshrrev_b32 v[vgprIndexU], 4, v[vgprSerial]

v_mul_u32_u24 v[vgprTmp], v[vgprIndexT], s[sgprSizeK]
v_mul_u32_u24 v[vgprOffset], 32, v[vgprIndexU]
v_add_u32 v[vgprOffset], v[vgprOffset], v[vgprTmp]
v_lshlrev_b32 v[vgprOffset], 2, v[vgprOffset] // * 4 bits per element
v_lshrrev_b32 v[vgprOffset], 3, v[vgprOffset] // / 8 bits per byte

buffer_load_dwordx4 v[vgprValueA+0:vgprValueA+3], v[vgprOffset], s[sgprSrcA:sgprSrcA+3], 0 offen offset:0
buffer_load_dwordx4 v[vgprValueB+0:vgprValueB+3], v[vgprOffset], s[sgprSrcB:sgprSrcB+3], 0 offen offset:0

v_and_b32 v[vgprIndexT], v[vgprSerial], 15
v_lshrrev_b32 v[vgprIndexU], 4, v[vgprSerial]

v_mul_u32_u24 v[vgprTmp], v[vgprIndexT], s[sgprSizeK]
v_lshrrev_b32 v[vgprTmp], 5, v[vgprTmp]
v_add_u32 v[vgprOffset], v[vgprIndexU], v[vgprTmp]

buffer_load_ubyte v[vgprValueSA], v[vgprOffset], s[sgprSrcSA:sgprSrcSA+3], 0 offen offset:0
buffer_load_ubyte v[vgprValueSB], v[vgprOffset], s[sgprSrcSB:sgprSrcSB+3], 0 offen offset:0

s_waitcnt vmcnt(0)

v_accvgpr_write acc0, 0x0
v_accvgpr_write acc1, 0x0
v_accvgpr_write acc2, 0x0
v_accvgpr_write acc3, 0x0

# v_mov_b32 v[vgprValueSA], 0x00000080
# v_mov_b32 v[vgprValueSB], 0x00000080

# v_mov_b32 v[vgprValueA+0], 0x22222222
# v_mov_b32 v[vgprValueA+1], 0x22222222
# v_mov_b32 v[vgprValueA+2], 0x22222222
# v_mov_b32 v[vgprValueA+3], 0x22222222

# v_mov_b32 v[vgprValueB+0], 0x22222222
# v_mov_b32 v[vgprValueB+1], 0x22222222
# v_mov_b32 v[vgprValueB+2], 0x22222222
# v_mov_b32 v[vgprValueB+3], 0x22222222

s_nop 7
s_nop 7
s_nop 7
s_nop 7

# v_mfma_ld_scale_b32 v[vgprValueSB], v[vgprValueSA]
# v_mfma_f32_16x16x128_f8f6f4 acc[0:3], v[vgprValueB+0:vgprValueB+3], v[vgprValueA+0:vgprValueA+3], acc[0:3] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[vgprValueB+0:vgprValueB+3], v[vgprValueA+0:vgprValueA+3], acc[0:3], v[vgprValueSB], v[vgprValueSA] cbsz:4 blgp:4
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7
s_nop 7

v_accvgpr_read_b32 v[vgprValueC+0], acc0
v_accvgpr_read_b32 v[vgprValueC+1], acc1
v_accvgpr_read_b32 v[vgprValueC+2], acc2
v_accvgpr_read_b32 v[vgprValueC+3], acc3
s_nop 7

v_and_b32 v[vgprIndexU], v[vgprSerial], 15
v_lshrrev_b32 v[vgprIndexT], 4, v[vgprSerial]
v_lshlrev_b32 v[vgprIndexT], 2, v[vgprIndexT]
v_mul_u32_u24 v[vgprOffset], v[vgprIndexT], s[sgprSizeM]
v_add_u32 v[vgprOffset], v[vgprOffset], v[vgprIndexU]
v_lshlrev_b32 v[vgprOffset], 2 v[vgprOffset]

v_lshlrev_b32 v[vgprTmp], 2, s[sgprSizeM]
buffer_store_dword v[vgprValueC+0], v[vgprOffset], s[sgprDstC:sgprDstC+3], 0 offen offset:0
v_add_u32 v[vgprOffset], v[vgprOffset], v[vgprTmp]
buffer_store_dword v[vgprValueC+1], v[vgprOffset], s[sgprDstC:sgprDstC+3], 0 offen offset:0
v_add_u32 v[vgprOffset], v[vgprOffset], v[vgprTmp]
buffer_store_dword v[vgprValueC+2], v[vgprOffset], s[sgprDstC:sgprDstC+3], 0 offen offset:0
v_add_u32 v[vgprOffset], v[vgprOffset], v[vgprTmp]
buffer_store_dword v[vgprValueC+3], v[vgprOffset], s[sgprDstC:sgprDstC+3], 0 offen offset:0
s_waitcnt vmcnt(0)

s_endpgm
.LMatMul_end:
.size MatMul, .LMatMul_end - MatMul
