// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

#include "gemmini_params.h"

#define GEMMINI_ASSERTIONS

// Accelerator interface
#ifdef __riscv
#include "rocc-software/src/xcustom.h"
#endif

// Counter Definition
#include "gemmini_counter.h"

#include "ggml-gemmini-args.h"
#include <gemmini/log.h>
#include <gemmini/cycle_reader.hpp>
#include <gemmini/layer.hpp>
#include "quants/stripe.hpp"
#include "quants/act/dispatch.hpp"
#include "quants/common/dequant.hpp"

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define k_LOOP_WS 8
#define k_LOOP_WS_CONFIG_BOUNDS 9
#define k_LOOP_WS_CONFIG_ADDRS_AB 10
#define k_LOOP_WS_CONFIG_ADDRS_DC 11
#define k_LOOP_WS_CONFIG_STRIDES_AB 12
#define k_LOOP_WS_CONFIG_STRIDES_DC 13

#define k_MVIN3 14

#define k_COUNTER 126

#define k_LOOP_CONV_WS 15
#define k_LOOP_CONV_WS_CONFIG_1 16
#define k_LOOP_CONV_WS_CONFIG_2 17
#define k_LOOP_CONV_WS_CONFIG_3 18
#define k_LOOP_CONV_WS_CONFIG_4 19
#define k_LOOP_CONV_WS_CONFIG_5 20
#define k_LOOP_CONV_WS_CONFIG_6 21

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_BERT 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#ifndef GEMMINI_KBLOCK_DEBUG
#define GEMMINI_KBLOCK_DEBUG 0
#endif
#ifndef GEMMINI_WS_DEBUG
#define GEMMINI_WS_DEBUG 0
#endif
#ifndef GEMMINI_WS_LOOP_DEBUG
#define GEMMINI_WS_LOOP_DEBUG 0
#endif
#ifndef GEMMINI_DISABLE_WS_REUSE
#define GEMMINI_DISABLE_WS_REUSE 0
#endif
#ifndef ACTIVATION_BLOCK_SCALE
#define ACTIVATION_BLOCK_SCALE 1
#endif

#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.b = x;
    return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.f = x;
    return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.b = x;
    return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.f = x;
    return un.b;
}

bool elem_t_isnan(elem_t x) {
    elem_t_bits bits = elem_t_to_elem_t_bits(x);
    uint64_t exp = (bits >> (ELEM_T_SIG_BITS-1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
    acc_t_bits bits = acc_t_to_acc_t_bits(x);
    uint64_t exp = (bits >> (ACC_T_SIG_BITS-1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}
#endif

#ifdef HAS_MVIN_SCALE
static scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}
#else
#define scale_t_to_scale_t_bits(x) 0
#endif

#ifdef HAS_MVIN_ACC_SCALE
static scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

static acc_scale_t acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.f = x;
    return un.b;
}

#ifdef __riscv
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)
#else
#define ROCC_INSTRUCTION(...)         do { } while (0);
#define ROCC_INSTRUCTION_RS1_RS2(...) do { } while (0);
#define ROCC_INSTRUCTION_R_R_R(...)   do { } while (0);
#define ROCC_INSTRUCTION_R_R_I(...)   do { } while (0);
#define ROCC_INSTRUCTION_R_R(...)     do { } while (0);
#define ROCC_INSTRUCTION_RW_R(...)    do { } while (0);
#define ROCC_INSTRUCTION_RS(...)      do { } while (0);
#define ROCC_INSTRUCTION_S(...)       do { } while (0);
#define ROCC_INSTRUCTION_0_R_R(...)   do { } while (0);
#endif

// mvin and mvout
#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD) \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD) \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// config
#define gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG); \

#define gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false)

#define gemmini_extended_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

#define gemmini_config_ex(dataflow, sys_act, sys_shift) \
    gemmini_extended_config_ex(dataflow, sys_act, sys_shift, 1, 0, 0)

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define gemmini_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, 1, id) \

#define gemmini_extended3_config_ld(stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(stride, scale, shrunk, DIM, id)

#define gemmini_extended2_config_ld(stride, scale, shrunk) \
  gemmini_extended3_config_ld(stride, scale, shrunk, 0)

#define gemmini_extended_config_ld(stride, scale) \
  gemmini_extended2_config_ld(stride, scale, false)

#define gemmini_config_ld(stride) \
  gemmini_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

#define gemmini_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((uint64_t)(acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define gemmini_extended_config_st(stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_config_st(stride) \
    gemmini_extended_config_st(stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

#define gemmini_config_norm(q_const, q_const_type, set_stats_id_only, act_msb, stat_id, igelu_qb, igelu_qc) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (((uint64_t) ((uint32_t) q_const)) << 32) | ((q_const_type & 1) << 18) | ((set_stats_id_only & 1) << 17) | ((act_msb & 1) << 16) | ((uint64_t)stat_id << 8) | CONFIG_BERT, ((uint64_t)((uint32_t)(igelu_qc)) << 32) | ((uint64_t)((uint32_t)(igelu_qb))), k_CONFIG)

// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#ifdef __riscv
#define gemmini_fence() asm volatile("fence")
#else
#undef gemmini_fence
#define gemmini_fence() ((void)0)
#endif

#ifdef __riscv
// Counter access
#define gemmini_counter_access(rd, config_reg) \
  { \
    uint32_t _placeholder; \
    ROCC_INSTRUCTION(XCUSTOM_ACC, rd, config_reg, _placeholder, k_COUNTER) \
  }

// Read counter
static uint32_t counter_read(size_t index) {
  uint32_t config_reg = (index & 0x7) << 4;
  uint32_t res;
  gemmini_counter_access(res, config_reg);
  return res;
}

// Configure counter to take a new signal
static void counter_configure(size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0x7) << 4 | 0x8 | (counter_code & 0x3f) << 12 | non_incremental << 31;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

// Take a snapshot
static void counter_snapshot_take() {
  uint32_t config_reg = 0x4;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

// Counter snapshot reset
static void counter_snapshot_reset() {
  uint32_t config_reg = 0x2;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

// Counter module reset
static void counter_reset() {
  uint32_t config_reg = 0x1;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}
#else
#define gemmini_counter_access(...) do { } while (0)
static inline uint32_t counter_read(size_t) { return 0; }
static inline void counter_configure(size_t, size_t) {}
static inline void counter_snapshot_take() {}
static inline void counter_snapshot_reset() {}
static inline void counter_reset() {}
#endif

// modified by Youngshin --make ceil_divide_int to static inline func
static inline int ceil_divide_int(int a, int b){
    int c = (a % b == 0) ? ((int)(a/b)) :(((int)(a/b)) + 1); 
    if(a < b) c = 1;
    return c;
}

// weight-stationary matmul loop
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// weight-stationary conv loop
#define gemmini_loop_conv_ws(batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, dw, a_spad_id, b_spad_id) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 21) | ((uint64_t)(pdpad) << 10) | (uint64_t)(kernel_dilation), \
      ((uint64_t)(in_stride) << 48) | ((uint64_t)(weight_stride) << 32) | ((uint64_t)(out_stride) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, weights, \
      output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, bias, \
      input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(max_pixels_per_row) << 8) | ((dw) << 6) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }

// Tiling functions
static void sp_tiled_matmul_os(const elem_t * A, const elem_t * B, const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3));

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    gemmini_extended_config_ld(D_stride, D_scale_factor);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J-j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);

        gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  // Move-in B
  gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
    }
  }

  // Move-in A
  gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
    }
  }

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);

        if (k == 0) { // First iteration
          gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        } else { // All other iterations
          gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + j)*DIM*sizeof_C;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
      }
    }
  }
}


static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {
/*
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
    (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }
  for (size_t k = 0; k < K; k++) {
    for (size_t j = 0; j < J; j++) {
      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
        // Mvin A
        if (a_transpose) {
          if (j == 0 && i % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
            const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
            const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
          }
        } else {
          if (j == 0 && k % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
            const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
          }
        }
        // Mvin B
        if (b_transpose) {
          if (i == 0 && k % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
            const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (j == J-1 ? pad_J : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
          }
        } else {
          if (i == 0 && j % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
            const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
            const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
          }
        }
        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;
          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }
          const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          }
        }
        if (C != NULL && k == K-1) {
          // Move-out C (if not normalizing)
          if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
            const size_t rounded_j = (j / C_blocks) * C_blocks;
            const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
            void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;
            const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
            const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
            gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
          }
          // Move-out C (if normalizing)
          if (act == LAYERNORM && j == J - 1) {
            uint32_t norm_cmds[][2] = {{1,2},{3,4},{0,0}};
            const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
              const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                NORM_STAT_IDS : rows - row;
              for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                  gemmini_config_norm(0, 0, 0, 0, stat_id, 0, 0);
                  const size_t r = row + stat_id;
                  for (size_t jj = 0; jj < J; jj += C_blocks) {
                    uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                    if (jj + C_blocks >= J) {
                      norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                    } else {
                      norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                    }
                    void * const C_dram_addr = (int8_t*)C +
                      (i*C_row_stride + jj) * DIM * sizeof_C +
                      r * C_row_stride * sizeof_C;
                    const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                    const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                    gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                  }
                }
              }
            }
          } else if (act == SOFTMAX && j == J - 1) {
            uint32_t norm_cmds[][2] = {{5,5},{6,7},{0,0}};
            const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
              const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                NORM_STAT_IDS : rows - row;
              for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                  // set stat id only
                  gemmini_config_norm(0, 0, 1, 0, stat_id, 0, 0);
                  const size_t r = row + stat_id;
                  for (size_t jj = 0; jj < J; jj += C_blocks) {
                    uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                    if (jj + C_blocks >= J) {
                      norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                    } else {
                      norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                    }
                    void * const C_dram_addr = (int8_t*)C +
                      (i*C_row_stride + jj) * DIM * sizeof_C +
                      r * C_row_stride * sizeof_C;
                    const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                    const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                    gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
*/

  // Combined loop
  gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
    A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
    a_transpose, b_transpose,
    full_C, low_D, !no_bias || D == NULL,
    act, a_spad_id, b_spad_id, false);
}


static void tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
  gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
  gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;
    const acc_scale_t S = bert_scale;
    const acc_scale_t S_erf = (-0.2888 * ((S*S) / 2));

    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / S_erf;

    gemmini_config_norm(0, 0, 0, 0, 0, qb, qc);
  }

  if (act == SOFTMAX) {
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int) (0.693147 / bert_scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / bert_scale;
    const acc_t qc = c / (a*bert_scale*bert_scale);

    gemmini_config_norm(qln2, 0, 0, 1, 0, qb, qc);
    gemmini_config_norm(qln2_inv, 1, 0, 1, 0, qb, qc);
  }

  void (*inner)(const elem_t *, const elem_t *, const void *, void *,
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        int, int, int);

  if (dataflow == OUTPUT_STATIONARY) {
    inner = &sp_tiled_matmul_os;
  } else /* if (dataflow == WEIGHT_STATIONARY) */ {
    inner = &sp_tiled_matmul_ws;
  }

  // reuse operand if it fits scratchpad
  int a_spad_id = 0;
  int b_spad_id = 0;
  bool b_reuse = (J0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
  bool a_reuse = (I0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
#if GEMMINI_DISABLE_WS_REUSE
  if (dataflow == WEIGHT_STATIONARY) {
    a_reuse = false;
    b_reuse = false;
  }
#endif

#if GEMMINI_WS_DEBUG
  if (dataflow == WEIGHT_STATIONARY) {
      gemmini_log_debug(
        "[ws.outer] dim=(%zu,%zu,%zu) tile=(%zu,%zu,%zu) iter=(%zu,%zu,%zu) pad=(%zu,%zu,%zu) "
        "stride=(%zu,%zu,%zu,%zu) reuse=(A:%d,B:%d) flags=(aT:%d,bT:%d,fullC:%d,lowD:%d,noBias:%d,repeatBias:%d,act:%d)",
        dim_I, dim_J, dim_K,
        tile_I, tile_J, tile_K,
        I0, J0, K0,
        padding_I, padding_J, padding_K,
        stride_A, stride_B, stride_D, stride_C,
        a_reuse ? 1 : 0, b_reuse ? 1 : 0,
        a_transpose ? 1 : 0, b_transpose ? 1 : 0,
        full_C ? 1 : 0, low_D ? 1 : 0,
        no_bias ? 1 : 0, repeating_bias ? 1 : 0,
        act);
  }
#endif

  size_t ws_call_idx = 0;

#if GEMMINI_WS_LOOP_DEBUG && defined(__riscv)
  const bool ws_loop_debug = dataflow == WEIGHT_STATIONARY;
  uint64_t ws_loop_start = 0;
  if (ws_loop_debug) {
    counter_configure(0, LOAD_ACTIVE_CYCLE);
    counter_configure(1, EXE_ACTIVE_CYCLE);
    counter_configure(2, STORE_ACTIVE_CYCLE);
    counter_configure(3, LOOP_MATMUL_ACTIVE_CYCLES);
    counter_snapshot_reset();
    ws_loop_start = ggml::gemmini::cycle::read();
  }
#endif

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        if(a_reuse)
          a_spad_id = ((i0+k0) == 0) ? 1 : 2;
        if(b_reuse)
          b_spad_id = ((j0+k0) == 0) ? 1 : 2;

        const void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
        }

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

        const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

        const elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

        if(a_reuse && j0 >= 1) a = NULL;
        if(b_reuse && i0 >= 1) b = NULL;

#if GEMMINI_WS_DEBUG
        if (dataflow == WEIGHT_STATIONARY) {
          gemmini_log_debug(
              "[ws.inner.begin] idx=%zu i0=%zu j0=%zu k0=%zu IJK=(%zu,%zu,%zu) pad=(%zu,%zu,%zu) "
              "ptr=(A:%p,B:%p,pre:%p,out:%p) spad=(%d,%d) reuse=(A:%d,B:%d)",
              ws_call_idx, i0, j0, k0,
              I, J, K,
              pad_I, pad_J, pad_K,
              (const void *)a, (const void *)b, pre, out,
              a_spad_id, b_spad_id,
              a_reuse ? 1 : 0, b_reuse ? 1 : 0);
        }
#endif
        //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, a, b);
        (*inner)(a, b, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            act, a_spad_id, b_spad_id);

#if GEMMINI_WS_DEBUG
        if (dataflow == WEIGHT_STATIONARY) {
          gemmini_log_debug("[ws.inner.end] idx=%zu i0=%zu j0=%zu k0=%zu", ws_call_idx, i0, j0, k0);
        }
#endif
        ++ws_call_idx;
      }

  gemmini_fence();

#if GEMMINI_WS_LOOP_DEBUG && defined(__riscv)
  if (ws_loop_debug) {
    const uint64_t ws_loop_end = ggml::gemmini::cycle::read();
    counter_snapshot_take();
    gemmini_log_ws_loop(
        ws_loop_end - ws_loop_start,
        counter_read(0), counter_read(1), counter_read(2), counter_read(3),
        dim_I, dim_J, dim_K,
        tile_I, tile_J, tile_K,
        I0, J0, K0,
        a_reuse ? 1 : 0, b_reuse ? 1 : 0);
  }
#endif
}


static acc_t int_sqrt(acc_t n) {
  if (n == 0) return 0;

  int bits = 0;
  for (acc_t x = n; x > 0; x /= 2)
    bits++;

  acc_t x_prev = 1 << ((bits + 1) / 2);

  while (1) {
    acc_t x_next = (x_prev + n / x_prev) / 2;
    if (x_next >= x_prev) return x_prev;
    x_prev = x_next;
  };
}


static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale) {
  // Apply I-GELU if needed
  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;

    x = q * (q_erf + q1);
  }

  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU) {
    x = x < 0 ? 0 : x;
  }
  return x;
}

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define GEMMINI_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define GEMMINI_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define GEMMINI_ACC_SCALE(x, scale) (x)
#endif

static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias) {

  const int no_bias = D == NULL;
  if (act != LAYERNORM && act != SOFTMAX && !transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, bert_scale);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, bert_scale);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, bert_scale);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, bert_scale);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride

    // We also create a buffer that we can use for layernorms and softmaxes
    static acc_t c_buffer[1024];
    const size_t c_buffer_sz = sizeof(c_buffer)/sizeof(c_buffer[0]);
    if ((act == LAYERNORM || act == SOFTMAX) && DIM_J > c_buffer_sz) {
      printf("Matmul is too large to normalize\n");
      exit(1);
    }

    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }

        if (act == LAYERNORM || act == SOFTMAX)
          c_buffer[j] = sum;
        else
          *c = scale_and_sat(sum, act, scale, bert_scale);
      }

      if (act == LAYERNORM) {
        acc_t sum = 0;
        for (size_t j = 0; j < DIM_J; j++)
          sum += c_buffer[j];
        acc_t mean = sum / (acc_t)DIM_J;

        acc_t total_err_sq = 0;
        for (size_t j = 0; j < DIM_J; j++)
          total_err_sq += (c_buffer[j] - mean)*(c_buffer[j] - mean);
        acc_t variance = total_err_sq / (acc_t)DIM_J;

        acc_t stddev = int_sqrt(variance);
        if (variance == 0) stddev = 1;

        for (size_t j = 0; j < DIM_J; j++) {
          c_buffer[j] -= mean;
          c_buffer[j] /= stddev;

          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, scale, bert_scale);
        }
      } else if (act == SOFTMAX) {
        const scale_t a = 0.3585;
        const scale_t b = 1.353;
        const scale_t c = 0.344;

        // is SCALE supposed to be input scale?
        const acc_t qln2 = (acc_t) (0.693147 / bert_scale);
        const acc_t qln2_inv = 65536 / qln2;
        const acc_t qb = b / bert_scale;
        const acc_t qc = c / (a*bert_scale*bert_scale);

        // pass 1: get max_q
        acc_t max_q = -2147483648;
        for (size_t j = 0; j < DIM_J; j++) {
          if (c_buffer[j] > max_q) max_q = c_buffer[j];
        }

        // pass 2: calculate iexp(q_tilde) and sum(q_tilde)
        acc_t sum_exp = 0;
        for (size_t j = 0; j < DIM_J; j++) {
          acc_t q = c_buffer[j] - max_q;
          // [HS] matmul_cpu not used. It cannot be used for float
          acc_t z = 0;
          // acc_t z = (acc_t) (-q * qln2_inv) >> 16;
          acc_t qp = q + z * qln2;
          // [HS] also cannot be used for Float
          acc_t q_exp = 0;
          // acc_t q_exp = (qp + qb)*(qp + qb) + qc;
          c_buffer[j] = q_exp + z;
          // -- ends. Make some lines dummy.
          sum_exp += c_buffer[j];
        }

        // pass 3: divide by sum
        scale_t factor = (127.f) / (float) sum_exp; // what corresponds to 1 in output?
        for (size_t j = 0; j < DIM_J; j++) {
          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, factor, bert_scale);
        }
      }
    }
  }
}

namespace ggml { namespace gemmini {
  // FP32 matmul fallback; dimension/transpose conventions follow matmul_cpu
  static void matmul_cpu_fp(bool transA, bool transB,
                          size_t DIM_I, size_t DIM_J, size_t DIM_K,
                          const float* A, const float* B, const float* D,
                          float* C,
                          size_t stride_A, size_t stride_B,
                          size_t stride_D, size_t stride_C) {
  const size_t A_i_stride = transA ? 1        : stride_A;
  const size_t A_k_stride = transA ? stride_A : 1;
  const size_t B_j_stride = transB ? stride_B : 1;
  const size_t B_k_stride = transB ? 1        : stride_B;

  if (DIM_K == 0) {
    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        C[i * stride_C + j] = D == NULL ? 0.0f : D[i * stride_D + j];
      }
    }
    return;
  }

  const size_t I4 = DIM_I & ~(size_t)3;
  const size_t J4 = DIM_J & ~(size_t)3;

  // Main 4x4 tile loop
  for (size_t i = 0; i < I4; i += 4) {
    for (size_t j = 0; j < J4; j += 4) {
      float r00 = 0.0f, r01 = 0.0f, r02 = 0.0f, r03 = 0.0f;
      float r10 = 0.0f, r11 = 0.0f, r12 = 0.0f, r13 = 0.0f;
      float r20 = 0.0f, r21 = 0.0f, r22 = 0.0f, r23 = 0.0f;
      float r30 = 0.0f, r31 = 0.0f, r32 = 0.0f, r33 = 0.0f;

      const float* a0 = A + (i + 0) * A_i_stride;
      const float* a1 = A + (i + 1) * A_i_stride;
      const float* a2 = A + (i + 2) * A_i_stride;
      const float* a3 = A + (i + 3) * A_i_stride;

      const float* b0 = B + (j + 0) * B_j_stride;
      const float* b1 = B + (j + 1) * B_j_stride;
      const float* b2 = B + (j + 2) * B_j_stride;
      const float* b3 = B + (j + 3) * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        const float av0 = *a0;
        const float av1 = *a1;
        const float av2 = *a2;
        const float av3 = *a3;

        const float bv0 = *b0;
        const float bv1 = *b1;
        const float bv2 = *b2;
        const float bv3 = *b3;

        r00 += av0 * bv0; r01 += av0 * bv1; r02 += av0 * bv2; r03 += av0 * bv3;
        r10 += av1 * bv0; r11 += av1 * bv1; r12 += av1 * bv2; r13 += av1 * bv3;
        r20 += av2 * bv0; r21 += av2 * bv1; r22 += av2 * bv2; r23 += av2 * bv3;
        r30 += av3 * bv0; r31 += av3 * bv1; r32 += av3 * bv2; r33 += av3 * bv3;

        a0 += A_k_stride;
        a1 += A_k_stride;
        a2 += A_k_stride;
        a3 += A_k_stride;

        b0 += B_k_stride;
        b1 += B_k_stride;
        b2 += B_k_stride;
        b3 += B_k_stride;
      }

      if (D != NULL) {
        const float* d0 = D + (i + 0) * stride_D + j;
        const float* d1 = D + (i + 1) * stride_D + j;
        const float* d2 = D + (i + 2) * stride_D + j;
        const float* d3 = D + (i + 3) * stride_D + j;

        r00 += d0[0]; r01 += d0[1]; r02 += d0[2]; r03 += d0[3];
        r10 += d1[0]; r11 += d1[1]; r12 += d1[2]; r13 += d1[3];
        r20 += d2[0]; r21 += d2[1]; r22 += d2[2]; r23 += d2[3];
        r30 += d3[0]; r31 += d3[1]; r32 += d3[2]; r33 += d3[3];
      }

      float* c0 = C + (i + 0) * stride_C + j;
      float* c1 = C + (i + 1) * stride_C + j;
      float* c2 = C + (i + 2) * stride_C + j;
      float* c3 = C + (i + 3) * stride_C + j;

      c0[0] = r00; c0[1] = r01; c0[2] = r02; c0[3] = r03;
      c1[0] = r10; c1[1] = r11; c1[2] = r12; c1[3] = r13;
      c2[0] = r20; c2[1] = r21; c2[2] = r22; c2[3] = r23;
      c3[0] = r30; c3[1] = r31; c3[2] = r32; c3[3] = r33;
    }

    // Right tail: rows are still 4-wide, columns are scalar
    for (size_t j = J4; j < DIM_J; j++) {
      float r0 = 0.0f;
      float r1 = 0.0f;
      float r2 = 0.0f;
      float r3 = 0.0f;

      const float* a0 = A + (i + 0) * A_i_stride;
      const float* a1 = A + (i + 1) * A_i_stride;
      const float* a2 = A + (i + 2) * A_i_stride;
      const float* a3 = A + (i + 3) * A_i_stride;
      const float* b = B + j * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        const float bv = *b;

        r0 += (*a0) * bv;
        r1 += (*a1) * bv;
        r2 += (*a2) * bv;
        r3 += (*a3) * bv;

        a0 += A_k_stride;
        a1 += A_k_stride;
        a2 += A_k_stride;
        a3 += A_k_stride;
        b += B_k_stride;
      }

      if (D != NULL) {
        r0 += D[(i + 0) * stride_D + j];
        r1 += D[(i + 1) * stride_D + j];
        r2 += D[(i + 2) * stride_D + j];
        r3 += D[(i + 3) * stride_D + j];
      }

      C[(i + 0) * stride_C + j] = r0;
      C[(i + 1) * stride_C + j] = r1;
      C[(i + 2) * stride_C + j] = r2;
      C[(i + 3) * stride_C + j] = r3;
    }
  }

  // Bottom tail: remaining rows, 4 columns at a time
  for (size_t i = I4; i < DIM_I; i++) {
    for (size_t j = 0; j < J4; j += 4) {
      float r0 = 0.0f;
      float r1 = 0.0f;
      float r2 = 0.0f;
      float r3 = 0.0f;

      const float* a = A + i * A_i_stride;
      const float* b0 = B + (j + 0) * B_j_stride;
      const float* b1 = B + (j + 1) * B_j_stride;
      const float* b2 = B + (j + 2) * B_j_stride;
      const float* b3 = B + (j + 3) * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        const float av = *a;

        r0 += av * (*b0);
        r1 += av * (*b1);
        r2 += av * (*b2);
        r3 += av * (*b3);

        a += A_k_stride;
        b0 += B_k_stride;
        b1 += B_k_stride;
        b2 += B_k_stride;
        b3 += B_k_stride;
      }

      if (D != NULL) {
        const float* d = D + i * stride_D + j;
        r0 += d[0];
        r1 += d[1];
        r2 += d[2];
        r3 += d[3];
      }

      float* c = C + i * stride_C + j;
      c[0] = r0;
      c[1] = r1;
      c[2] = r2;
      c[3] = r3;
    }

    // Bottom-right scalar tail
    for (size_t j = J4; j < DIM_J; j++) {
      float sum = 0.0f;

      const float* a = A + i * A_i_stride;
      const float* b = B + j * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        sum += (*a) * (*b);
        a += A_k_stride;
        b += B_k_stride;
      }

      if (D != NULL) {
        sum += D[i * stride_D + j];
      }

      C[i * stride_C + j] = sum;
    }
  }
}

  // INT8 matmul fallback. Keep output INT32
  static void matmul_cpu_int32(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
                               const elem_t *A, const elem_t *B, const acc_t *D,
                               void *C,
                               size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                               scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                               int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias,
                               bool full_C)
  {

    if (!full_C)
    {
      matmul_cpu(transA, transB, DIM_I, DIM_J, DIM_K,
                 A, B, D, (elem_t *)C,
                 stride_A, stride_B, stride_D, stride_C,
                 A_scale_factor, B_scale_factor, D_scale_factor,
                 act, scale, bert_scale, repeating_bias);
      return;
    }

    if (act != NO_ACTIVATION)
    {
      printf("[matmul_cpu_int32] Expected NO_ACTIVATION but got act=%d (full_C-only path)\n", act);
      exit(1);
    }

    const int no_bias = D == NULL;
    acc_t *C_acc = (acc_t *)C;

    if (!transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0)
    {
      for (size_t i = 0; i < DIM_I; i += 4)
      {
        for (size_t j = 0; j < DIM_J; j += 4)
        {
          acc_t result[4][4];

          for (size_t ii = 0; ii < 4; ii++)
            for (size_t jj = 0; jj < 4; jj++)
            {
              const size_t bias_row = repeating_bias ? 0 : i + ii;
              result[ii][jj] = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j + jj), D_scale_factor);
            }

          for (size_t k = 0; k < DIM_K; k++)
          {
            result[0][0] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[0][1] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[0][2] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[0][3] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
            result[1][0] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[1][1] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[1][2] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[1][3] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
            result[2][0] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[2][1] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[2][2] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[2][3] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
            result[3][0] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[3][1] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[3][2] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[3][3] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
          }

          for (size_t ii = 0; ii < 4; ++ii)
            for (size_t jj = 0; jj < 4; ++jj)
            {
              const size_t idx = (i + ii) * stride_C + (j + jj);
              C_acc[idx] = result[ii][jj];
            }
        }
      }
    }
    else
    {
      size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A};
      size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1};

      static acc_t c_buffer[1024];
      const size_t c_buffer_sz = sizeof(c_buffer) / sizeof(c_buffer[0]);
      if ((act == LAYERNORM || act == SOFTMAX) && DIM_J > c_buffer_sz)
      {
        printf("Matmul is too large to normalize\n");
        exit(1);
      }

      for (size_t i = 0; i < DIM_I; i++)
      {
        for (size_t j = 0; j < DIM_J; j++)
        {
          const size_t idx = (i * stride_C) + j;

          const size_t bias_row = repeating_bias ? 0 : i;
          acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);
          const acc_t bias_term = sum;

          for (size_t k = 0; k < DIM_K; k++)
          {
            const size_t a_off = i * A_dim_strides[0] + k * A_dim_strides[1];
            const size_t b_off = j * B_dim_strides[0] + k * B_dim_strides[1];
            const elem_t a_raw = A[a_off];
            const elem_t b_raw = B[b_off];
            const acc_t a_scaled = GEMMINI_SCALE(a_raw, A_scale_factor);
            const acc_t b_scaled = GEMMINI_SCALE(b_raw, B_scale_factor);
            const acc_t prod = a_scaled * b_scaled;
            sum += prod;
          }

          C_acc[idx] = sum;
        }
      }
    }
  }
}}

#undef GEMMINI_SCALE

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t : int {OS, WS, CPU}; // TODO rename this so it's name also applies to convs

// This function runs a tiled matrix mulctiplication, with hardcoded tiling
// factors
static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D)) {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX) {
    if (tiled_matmul_type == OS) {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J) {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
    tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_J, tile_K,
        act, scale, bert_scale, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        (int)tiled_matmul_type);
  } else /*if (tiled_matmul_type == CPU)*/ {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias);
  }
}

namespace ggml { namespace gemmini {
  // TODO: 김동현
  // tiled_matmul을 참고하여, fp 텐서를 입력받아 처리하도록 구현.
  static void tiled_matmul_outer_fp(size_t dim_I, size_t dim_J, size_t dim_K,
        const float* A, const float* B,
        const float* D, float* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        enum tiled_matmul_type_t tiled_matmul_type) {
    (void) dim_I; (void) dim_J; (void) dim_K;
    (void) A; (void) B; (void) D; (void) C;
    (void) stride_A; (void) stride_B; (void) stride_D; (void) stride_C;
    (void) tile_I; (void) tile_J; (void) tile_K;
    (void) transpose_A; (void) transpose_B; (void) tiled_matmul_type;
  }

  // tiled matrix mulctiplication fp x fp
  static void tiled_matmul_fp(size_t dim_I, size_t dim_J, size_t dim_K,
        const float* A, const float* B,
        const float* D, float* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        enum tiled_matmul_type_t tiled_matmul_type) {
    if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
      tiled_matmul_outer_fp(dim_I, dim_J, dim_K,
          A, B, D, C,
          stride_A, stride_B, stride_D, stride_C,
          tile_I, tile_J, tile_K,
          transpose_A, transpose_B,
          tiled_matmul_type);
    } else {
      matmul_cpu_fp(transpose_A, transpose_B, dim_I, dim_J, dim_K,
          A, B, D, C,
          stride_A, stride_B, stride_D, stride_C);
    }
  }

}}

static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}


static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}

namespace ggml { namespace gemmini {
  // fp x fp gemmini. baseline system
  // TODO: 김동현
  static void tiled_matmul_auto_fp(struct ggml_gemmini_args_t *args)
  {
    const char *layer = ggml::gemmini::types::to_string(args->layer_type);

    // tile size와 block size 매칭을 통해, tiled별 연산 결과를 dequantize해서 llama.cpp의 output으로 전달
    uint64_t start = ggml::gemmini::cycle::read();
    if (args == NULL)
      return;

    const size_t dim_I = args->I;
    const size_t dim_J = args->J;
    const size_t dim_K = args->K;

    const enum tiled_matmul_type_t tiled_matmul_type = args->tiled_matmul_type;
    const int act = args->act;

    size_t tile_I = args->tile_I;
    size_t tile_J = args->tile_J;
    size_t tile_K = args->tile_K;

    const bool cpu_fallback = tiled_matmul_type == CPU;

      start = ggml::gemmini::cycle::read();
      // auto tiling으로 선정된 K 타일을 block에 매칭하여 bloctile_k_K를 사용해 호출
      /* if (cpu_fallback)
      {
        matmul_cpu_fp(dim_I, dim_J, dim_K,
                           A_block, B_block, D_block, acc_ptr,
                           block_stride_A, block_stride_B, args->sD, dim_J,
                           1.0f, 1.0f, args->scale_D,
                           act, args->scale, args->bert_scale, args->repeating_bias,
                           tile_I, tile_J, bloctile_k_K,
                           args->transpose_A, args->transpose_B,
                           true, args->low_D,
                           args->weightA,
                           tiled_matmul_type);
      }
      else
      {
        tiled_matmul(dim_I, dim_J, dim_K,
                     A_block, B_block, D_block, acc_ptr,
                     block_stride_A, block_stride_B, args->sD, dim_J,
                     1.0f, 1.0f, args->scale_D,
                     act, args->scale, args->bert_scale, args->repeating_bias,
                     tile_I, tile_J, bloctile_k_K,
                     args->transpose_A, args->transpose_B,
                     args->full_C, args->low_D,
                     args->weightA,
                     tiled_matmul_type);
      } */
      uint64_t end = ggml::gemmini::cycle::read();
      gemmini_log_cycle(layer, "[tiled_matmul_auto_fp32] npu.Gemmini HW tiled_matmul", start, end);

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    exit(EXIT_SUCCESS);
#endif
#endif

  }

  namespace gemmini_detail {
    static inline float apply_activation_exponent(float value, int16_t e_t, int16_t m)
    {
      if (e_t == std::numeric_limits<int16_t>::min())
        return value;

      return std::ldexp(value, static_cast<int>(e_t) - static_cast<int>(m));
    }
  }

  static void gemmini_set_tile(struct ggml_gemmini_args_t *args){
    const char *layer = ggml::gemmini::types::to_string(args->layer_type);

    // tile size 계산
    uint64_t start = ggml::gemmini::cycle::read();
    if (args == NULL)
      return;

    const size_t dim_I = args->I;
    const size_t dim_J = args->J;
    const size_t dim_K = args->K;
    const enum tiled_matmul_type_t tiled_matmul_type = args->tiled_matmul_type;
    const int act = args->act;

        // gemmini 기본 auto tiling
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    // GEMMINI는 한 타일이 DIM(=16) 배수여야 하므로 논리 크기를 DIM에 맞춰 패딩
    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    // WS 모드에서는 스크래치패드와 ACC를 두 세트로 나눠 번갈아 쓰기 때문에 최대 사용량이 절반으로 감소
    const bool double_buffered = tiled_matmul_type == WS;

    // auto tiler가 탐색할 최대 scratchpad/ACC 행 수를 미리 계산
    const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 : BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

    // tile_I/tile_J/tile_K는 DIM 단위 매트릭스 개수(행렬 블록 수)
    size_t tile_I, tile_J, tile_K;

    if (act == LAYERNORM || act == SOFTMAX)
    {
      // 레이어정규화·소프트맥스는 누산 버퍼 전체에 한 행이 상주해야 하므로 I=1, K=1로 강제
      tile_I = 1;
      tile_J = dim_J_padded / DIM;
      tile_K = 1;
    }
    else if (double_buffered)
    {
      // WS 모드: scratchpad/ACC 용량이 절반이므로 db_* 상수로 계산한 최대치와 실제 필요량 중 작은 값을 선택
      tile_I = dim_I_padded / DIM < db_max_tile_i_j ? dim_I_padded / DIM : db_max_tile_i_j;
      tile_J = dim_J_padded / DIM < db_max_tile_i_j ? dim_J_padded / DIM : db_max_tile_i_j;
      tile_K = dim_K_padded / DIM < db_max_tile_k ? dim_K_padded / DIM : db_max_tile_k;
    }
    else
    {
      // OS 모드: 전체 scratchpad/ACC를 쓸 수 있으니 기본 max_* 한도와 비교
      tile_I = dim_I_padded / DIM < max_tile_i_j ? dim_I_padded / DIM : max_tile_i_j;
      tile_J = dim_J_padded / DIM < max_tile_i_j ? dim_J_padded / DIM : max_tile_i_j;
      tile_K = dim_K_padded / DIM < max_tile_k ? dim_K_padded / DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true)
    {
      bool increased = false;

      // J 타일을 한 칸 늘렸을 때 scratchpad/ACC 제약을 넘지 않고 실제 행렬도 수용 가능하면 확장
      if (tiled_matmul_total_spad_rows(tile_I, tile_J + 1, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I, tile_J + 1) <= max_acc_rows &&
          (tile_J + 1) * DIM <= dim_J_padded)
      {
        tile_J++;
        increased = true;
      }

      // I 타일 확장: scratchpad에는 A, ACC에는 C가 늘어나므로 두 조건을 모두 만족해야 함
      if (tiled_matmul_total_spad_rows(tile_I + 1, tile_J, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I + 1, tile_J) <= max_acc_rows &&
          (tile_I + 1) * DIM <= dim_I_padded)
      {
        tile_I++;
        increased = true;
      }

      // K 타일 확장: scratchpad만 추가로 필요하므로 accumulator 조건은 확인하지 않음
      if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K + 1) <= max_spad_rows &&
          (tile_K + 1) * DIM <= dim_K_padded)
      {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

    args->tile_I = tile_I;
    args->tile_J = tile_J;
    args->tile_K = tile_K;

    // tile size 디버깅
    gemmini_log_debug_layer(layer, "[set_tile] dim=(%zu,%zu,%zu) tiles=(%zu,%zu,%zu)",
                   dim_I, dim_J, dim_K, tile_I, tile_J, tile_K);
  }

  static inline int resolve_env_threads(const char *name, int fallback) {
    if (const char *env = getenv(name))
    {
      char *end = nullptr;
      const long parsed = strtol(env, &end, 10);
      if (end != env && end && *end == '\0' && parsed > 0)
        return static_cast<int>(std::min(parsed, static_cast<long>(INT_MAX)));
    }

    return fallback;
  }

  static inline int resolve_im2p_threads(size_t tile_pair_count) {
    int im2p_threads = 1;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const size_t capped_count = std::min(tile_pair_count, static_cast<size_t>(INT_MAX));
    im2p_threads = std::min(std::max(1, static_cast<int>(capped_count)), omp_get_max_threads());
#endif

    im2p_threads = resolve_env_threads("GEMMINI_OPENMP_THREADS", im2p_threads);
    im2p_threads = resolve_env_threads("IM2P_THREADS", im2p_threads);

    return std::max(1, im2p_threads);
  }

  static inline int resolve_exsia_threads(size_t output_count) {
    int exsia_threads = 1;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const size_t capped_count = std::min(output_count, static_cast<size_t>(INT_MAX));
    exsia_threads = std::min(std::max(1, static_cast<int>(capped_count)), omp_get_max_threads());
#endif

    exsia_threads = resolve_env_threads("GEMMINI_OPENMP_THREADS", exsia_threads);
    exsia_threads = resolve_env_threads("EXSIA_THREADS", exsia_threads);

    return std::max(1, exsia_threads);
  }

  static void tiled_matmul_im2p_impl(struct ggml_gemmini_args_t *args, bool use_cpu_dot, const char *cycle_label) {
    if (args == NULL)
      return;

    const char *layer = ggml::gemmini::types::to_string(args->layer_type);
    const bool native_q8 = args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1 ||
        args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
        args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    // tile size와 block size 매칭을 통해, tiled별 연산 결과를 dequantize해서 llama.cpp의 output으로 전달
    uint64_t start = ggml::gemmini::cycle::read();

    if (args->A == nullptr || args->f_out == nullptr)
      return;

    if (!native_q8 && (args->B == nullptr || args->c_b == nullptr))
      return;

    if (!native_q8)
    {
      const size_t stripe_J_check = args->stripe_J_or_rowwise_elems();
      if (stripe_J_check > 1)
      {
        if (args->s_rf_stripe == nullptr || args->R_stripe == nullptr)
          return;
      }
      else
      {
        if (args->s_rf == nullptr || args->R == nullptr)
          return;
      }
    }

    if (args->tile_I == 0 || args->tile_J == 0 || args->tile_K == 0)
      gemmini_set_tile(args);

    const size_t dim_I = args->I;
    const size_t dim_J = args->J;
    const size_t dim_K = args->K;
    if (dim_I == 0 || dim_J == 0 || dim_K == 0 || args->blocks_per_row == 0)
      return;

    GGML_ASSERT(args->act == NO_ACTIVATION);

    size_t block_size_k = args->block_size_k > 0
                              ? static_cast<size_t>(args->block_size_k)
                              : static_cast<size_t>(QK8_0);
    if (args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2)
      block_size_k = static_cast<size_t>(QK8_H2);
    else if (args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
             args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2)
      block_size_k = static_cast<size_t>(QK8_HP);
    if (block_size_k == 0)
      return;

    const size_t tile_I = args->tile_I;
    const size_t tile_J = args->tile_J;
    const size_t tile_K = args->tile_K;
    const size_t out_row_stride = args->stride_f_out ? args->stride_f_out : dim_J;
    const size_t out_col_stride = args->col_stride_f_out ? args->col_stride_f_out : 1;
    const size_t scale_rows = args->blocks_J > 0 ? args->blocks_J : dim_J;
    const size_t stripe_J = args->stripe_J_or_rowwise_elems();
    const bool use_stripe_scales = (stripe_J > 1) && (args->s_rf_stripe != nullptr) && (args->R_stripe != nullptr);
    const size_t num_i_tiles = (dim_I + tile_I * DIM - 1) / (tile_I * DIM);
    const size_t num_j_tiles = (dim_J + tile_J * DIM - 1) / (tile_J * DIM);
    const size_t num_tile_ks = (dim_K + tile_K * DIM - 1) / (tile_K * DIM);
    const auto *activation_meta = std::get_if<ggml::gemmini::quants::act::exsia::Meta>(&args->act_quant.storage());
    const auto *tensor_meta = std::get_if<ggml::gemmini::quants::act::tensor::Meta>(&args->act_quant.storage());
    const auto *token_meta = std::get_if<ggml::gemmini::quants::act::token::Meta>(&args->act_quant.storage());
    const auto *stripe_meta = std::get_if<ggml::gemmini::quants::act::stripe::Meta>(&args->act_quant.storage());

    if (stripe_meta != nullptr)
    {
      const size_t rows_per_stripe = tile_I * DIM;
      const size_t expected_scale_count = (dim_I + rows_per_stripe - 1) / rows_per_stripe;
      if (stripe_meta->scales.size() != expected_scale_count)
      {
        gemmini_log_debug_layer(layer,
            "[tiled_block_matmul_auto] reject STRIPE activation scale cardinality: got=%zu expected=%zu",
            stripe_meta->scales.size(), expected_scale_count);
        return;
      }
      for (const float scale : stripe_meta->scales)
      {
        if (!std::isfinite(scale) || scale <= 0.0f)
        {
          gemmini_log_debug_layer(layer,
              "[tiled_block_matmul_auto] reject STRIPE activation scale value: scale=%g",
              static_cast<double>(scale));
          return;
        }
      }
    }

    if (native_q8)
    {
      bool valid_format = false;
      switch (args->weight_format) {
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_h1:
          valid_format = args->has_q8_h1_im2p_contract();
          break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_h2:
          valid_format = args->has_q8_h2_im2p_contract();
          break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1:
          valid_format = args->has_q8_hp1_im2p_contract();
          break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2:
          valid_format = args->has_q8_hp2_im2p_contract();
          break;
        default:
          valid_format = false;
          break;
      }
      if (!valid_format)
      {
        gemmini_log_debug_layer(layer, "[tiled_block_matmul_auto] reject native Q8 contract: format=%d", static_cast<int>(args->weight_format));
        return;
      }
    }

#if !ERROR_COMPENSATION
    std::vector<int32_t> activation_residuals;
    if (activation_meta != nullptr && !activation_meta->outliers.empty())
    {
      if (dim_I > std::numeric_limits<size_t>::max() / dim_K)
        return;

      activation_residuals.assign(dim_I * dim_K, 0);
      for (const auto &outlier : activation_meta->outliers)
      {
        if (outlier.row < 0 || outlier.col < 0)
          continue;

        const size_t row = static_cast<size_t>(outlier.row);
        const size_t col = static_cast<size_t>(outlier.col);
        if (row < dim_I && col < dim_K)
          activation_residuals[row * dim_K + col] += outlier.residual;
      }
    }
#endif

    if (activation_meta != nullptr)
    {
      const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
      for (size_t tile_i_idx = 0; tile_i_idx < num_i_tiles; ++tile_i_idx)
      {
        if (activation_meta->resolve_stripe_theta(static_cast<int>(tile_i_idx)) == invalid_theta)
          return;
      }
    }

    for (size_t i = 0; i < dim_I; ++i)
    {
      float *row_out = args->f_out + i * out_row_stride;
      for (size_t j = 0; j < dim_J; ++j)
        row_out[j * out_col_stride] = 0.0f;
    }

    if (use_stripe_scales && !args->stripe_mode_matches_tile_j(tile_J * DIM))
    {
      gemmini_log_debug_layer(layer,
          "[tiled_block_matmul_auto] reject stripe mode contract: stripe_J=%zu tile_J_elems=%zu",
          args->stripe_J, tile_J * DIM);
      return;
    }

    gemmini_log_debug_layer(layer,
        "[tiled_block_matmul_auto] dim=(I=%zu,J=%zu,K=%zu) tile_size: A=(I=%zu x K=%zu) B=(J=%zu x K=%zu) | num_tiles: I=%zu J=%zu K=%zu blocks_per_row=%zu stripe_J=%zu",
        dim_I, dim_J, dim_K, tile_I * DIM, tile_K * DIM, tile_J * DIM, tile_K * DIM, num_i_tiles, num_j_tiles, num_tile_ks, args->blocks_per_row, stripe_J);

    const elem_t *A = args->A;
    const elem_t *B = args->B;
    const size_t stride_A = args->sA ? args->sA : dim_K;
    const size_t row_stride_B = args->sB;
    if (!native_q8 && row_stride_B == 0)
      return;

    const int16_t activation_e_s = activation_meta ? activation_meta->e_s : 0;
    const int16_t activation_rho = activation_meta ? activation_meta->rho : 0;
    const size_t tile_pair_count = num_i_tiles * num_j_tiles;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const bool use_im2p_openmp = use_cpu_dot && tile_pair_count > 1;
    const int im2p_threads = use_im2p_openmp ? resolve_im2p_threads(tile_pair_count) : 1;
    if (use_im2p_openmp)
      gemmini_log_debug_layer(layer,
          "[tiled_matmul_im2p_sw] openmp tile_pairs=%zu threads=%d",
          tile_pair_count, im2p_threads);
#else
    const bool use_im2p_openmp = false;
#endif

    auto run_tile_pair = [&](size_t tile_i_idx, size_t tile_j_idx) {
      static thread_local std::vector<elem_t> a_tile;
      static thread_local std::vector<int32_t> a_tile_i32;
      static thread_local std::vector<int64_t> acc32;
      static thread_local std::vector<double> acc_fp;

      const size_t tile_i = tile_i_idx * tile_I * DIM;
      const size_t tile_j = tile_j_idx * tile_J * DIM;
      const size_t tile_i_actual = std::min(tile_I * DIM, dim_I - tile_i);
      const size_t tile_j_actual = std::min(tile_J * DIM, dim_J - tile_j);
      const int tile_row_idx = static_cast<int>(tile_i_idx);
      const size_t stripe_col_idx = tile_j_idx;

      if (!use_im2p_openmp)
      {
        gemmini_log_debug_layer(layer,
            "[preload] weight stripe: tile_j=%zu tile_j_actual=%zu stripe_col=%zu stripe_J=%zu use_stripe=%s",
            tile_j, tile_j_actual, stripe_col_idx, stripe_J,
            use_stripe_scales ? "yes" : "no");
      }

      if (native_q8)
        acc_fp.assign(tile_i_actual * tile_j_actual, 0.0);
      else
        acc32.assign(tile_i_actual * tile_j_actual, 0);

      for (size_t tile_k = 0; tile_k < dim_K; tile_k += tile_K * DIM)
      {
        const size_t tile_k_actual = std::min(tile_K * DIM, dim_K - tile_k);
        const size_t tile_k_padded = ((tile_k_actual + block_size_k - 1) / block_size_k) * block_size_k;
        const size_t wavefront = static_cast<size_t>(tile_row_idx) + stripe_col_idx;
        if (!use_im2p_openmp)
        {
          gemmini_log_debug_layer(layer,
              "[wavefront] wave=%zu tile=(row=%d,col=%zu) stripe=(I=%zu,J=%zu)",
              wavefront, tile_row_idx, stripe_col_idx,
              tile_i_actual, tile_j_actual);
        }

        if (native_q8)
          a_tile_i32.assign(tile_i_actual * tile_k_padded, 0);
        else
          a_tile.assign(tile_i_actual * tile_k_padded, 0);

        for (size_t i = 0; i < tile_i_actual; ++i)
        {
          for (size_t kk = 0; kk < tile_k_actual; ++kk)
          {
            const size_t src_i = tile_i + i;
            const size_t src_k = tile_k + kk;
            int32_t a = static_cast<int32_t>(args->transpose_A
                                                 ? A[src_k * stride_A + src_i]
                                                 : A[src_i * stride_A + src_k]);
#if !ERROR_COMPENSATION
            if (!activation_residuals.empty())
              a += activation_residuals[src_i * dim_K + src_k];
#endif
            if (native_q8)
              a_tile_i32[i * tile_k_padded + kk] = a;
            else
              a_tile[i * tile_k_padded + kk] = static_cast<elem_t>(a);
          }
        }
        if (!use_im2p_openmp)
        {
          gemmini_log_debug_layer(layer,
              "[prefetch] activation tile: tile_i=%zu tile_k=%zu size=(%zu x %zu)",
              tile_i, tile_k, tile_i_actual, tile_k_actual);
        }

        std::vector<double> native_scale_lut;        // (j,weight_blk)당 1회 계산, i-루프 밖으로 hoist
        std::vector<const int8_t *> native_qs_lut;   // qs 포인터도 hoist: 포맷 무관 단일 핫 루프
        for (size_t k_block = 0; k_block < tile_k_actual;)
        {
          const size_t global_k = tile_k + k_block;
          const size_t k_in_weight_block = global_k % block_size_k;
          const size_t block_k_actual = std::min(block_size_k - k_in_weight_block, tile_k_actual - k_block);
          const size_t weight_blk = global_k / block_size_k;
          if (weight_blk >= args->blocks_per_row)
          {
            k_block += block_k_actual;
            continue;
          }

          if (native_q8)
          {
            native_scale_lut.assign(tile_j_actual, 0.0);
            native_qs_lut.assign(tile_j_actual, nullptr);
            for (size_t j = 0; j < tile_j_actual; ++j)
            {
              const size_t global_j = tile_j + j;
              if (global_j >= scale_rows) continue;
              switch (args->weight_format) {
                case ggml_gemmini_args_t::im2p_weight_format_t::q8_h1: {
                  const block_q8_h1 *b = args->q8_h1_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = static_cast<double>(b->s_rf) *
                      static_cast<double>(static_cast<uint32_t>(b->c_b) + static_cast<uint32_t>(b->R));
                      native_qs_lut[j] = b->qs; }
                  break; }
                case ggml_gemmini_args_t::im2p_weight_format_t::q8_h2: {
                  const block_q8_h2 *b = args->q8_h2_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = static_cast<double>(b->channel_scale) * static_cast<double>(b->m) / 255.0;
                      native_qs_lut[j] = b->qs; }
                  break; }
                case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1: {
                  const block_q8_hp1 *b = args->q8_hp1_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = b->m == INT16_MIN ? 0.0 : static_cast<double>(gemmini_ldexp_fast_pos(b->channel_scale, b->m));
                      native_qs_lut[j] = b->qs; }
                  break; }
                case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2: {
                  const block_q8_hp2 *b = args->q8_hp2_block(global_j, weight_blk);
                  if (b) { native_scale_lut[j] = b->m == INT16_MIN ? 0.0 : static_cast<double>(gemmini_ldexp_fast_pos(b->channel_scale, b->m));
                      native_qs_lut[j] = b->qs; }
                  break; }
                default: break;
              }
            }
          }

          for (size_t i = 0; i < tile_i_actual; ++i)
          {
            const elem_t *a_row = native_q8 ? nullptr : a_tile.data() + i * tile_k_padded + k_block;
            const int32_t *a_row_i32 = native_q8 ? a_tile_i32.data() + i * tile_k_padded + k_block : nullptr;
            int64_t *acc_row = native_q8 ? nullptr : acc32.data() + i * tile_j_actual;

            for (size_t j = 0; j < tile_j_actual; ++j)
            {
              const size_t global_j = tile_j + j;
              if (global_j >= scale_rows)
                continue;

              const elem_t *b_block = native_q8 ? nullptr : B + global_j * row_stride_B + global_k;
              int32_t block_dot = 0;
              double native_block_sum = 0.0;
              if (native_q8)
              {
                const int8_t *qs = native_qs_lut[j];
                if (qs == nullptr)
                  continue;

                const double weight_scale = native_scale_lut[j];
                for (size_t kk = 0; kk < block_k_actual; ++kk)
                  block_dot += a_row_i32[kk] * static_cast<int32_t>(qs[k_in_weight_block + kk]);
                native_block_sum = static_cast<double>(block_dot) * weight_scale;

                acc_fp[i * tile_j_actual + j] += native_block_sum;
                continue;
              }

              if (use_cpu_dot)
              {
                acc_t block_dot_acc = 0;
                matmul_cpu_int32(false, false, 1, 1, block_k_actual,
                    a_row, b_block, nullptr, &block_dot_acc,
                    block_k_actual, 1, 0, 1,
                    1, 1, 1,
                    NO_ACTIVATION, 0, 0, false, true);
                block_dot = static_cast<int32_t>(block_dot_acc);
              }
              else
              {
                for (size_t kk = 0; kk < block_k_actual; ++kk)
                  block_dot += static_cast<int32_t>(a_row[kk]) * static_cast<int32_t>(b_block[kk]);
              }

              const int64_t acc_before = acc_row[j];
              const size_t scale_idx = global_j * args->blocks_per_row + weight_blk;
              const uint64_t c_eff = static_cast<uint64_t>(static_cast<uint16_t>(args->c_b[scale_idx])) +
                  (use_stripe_scales
                      ? static_cast<uint64_t>(args->R_stripe[global_j / stripe_J])
                      : static_cast<uint64_t>(args->R[global_j]));
              acc_row[j] += static_cast<int64_t>(block_dot) * static_cast<int64_t>(c_eff);

              if (!use_im2p_openmp && i == 0 && j == 0)
              {
                  gemmini_log_debug_layer(layer,
                      "[tiled_block_matmul_auto] block-dequant i=%zu j=%zu k=%zu weight_blk=%zu "
                    "dot=%d c_b=%u R=%u c_eff=%llu acc_before=%ld acc_after=%ld stripe=%s",
                    tile_i + i, global_j, tile_k + k_block, weight_blk,
                    block_dot,
                    static_cast<unsigned int>(args->c_b[scale_idx]),
                    use_stripe_scales ? static_cast<unsigned int>(args->R_stripe[global_j / stripe_J]) : static_cast<unsigned int>(args->R[global_j]),
                    static_cast<unsigned long long>(c_eff),
                    static_cast<long>(acc_before),
                    static_cast<long>(acc_row[j]),
                    use_stripe_scales ? "1" : "0");
              }
            }
          }

          k_block += block_k_actual;
        }

        if (!use_im2p_openmp)
        {
          gemmini_log_debug_layer(layer,
               "[tiled_block_matmul_auto] iter I=[%zu:%zu) J=[%zu:%zu) K=[%zu:%zu) K_pad=%zu blk=[%zu:%zu) pad=%zu e_t=%d rho=%d q=%d",
              tile_i, tile_i + tile_i_actual, tile_j, tile_j + tile_j_actual,
              tile_k, tile_k + tile_k_actual, tile_k_padded,
              tile_k / block_size_k, (tile_k + tile_k_actual + block_size_k - 1) / block_size_k,
              tile_k_padded - tile_k_actual,
              static_cast<int>(activation_e_s),
               static_cast<int>(activation_rho),
              0);
        }
      }

      const int16_t tile_activation_e_s = activation_meta ? activation_meta->resolve_stripe_theta(tile_row_idx) : 0;
      const float activation_scale = stripe_meta
          ? stripe_meta->scales[tile_i_idx]
          : (activation_meta
                ? gemmini_detail::apply_activation_exponent(1.0f, tile_activation_e_s, 0)
                : (tensor_meta ? tensor_meta->scale : 1.0f));
      const ggml::gemmini::quants::Stripe output_stripe(tile_i_actual, tile_j_actual, tile_i, tile_j);
      float *stripe_out = args->f_out + tile_i * out_row_stride + tile_j * out_col_stride;

      if (!native_q8 && !use_im2p_openmp && tile_i_actual > 0 && tile_j_actual > 0 && tile_j < scale_rows)
      {
        const float s_rf_val = use_stripe_scales ? args->s_rf_stripe[stripe_col_idx] : args->s_rf[tile_j];
        const float acc_fp = static_cast<float>(static_cast<double>(acc32[0]) * static_cast<double>(s_rf_val));
        const float contrib = acc_fp * activation_scale;
        gemmini_log_debug_layer(layer,
            "[tiled_block_matmul_auto] stripe-dequant i=%zu j=%zu "
            "acc=%ld s_rf=%.9f acc*s_rf=%.6f exp=%.6f contrib=%.6f -> f_out+=%.6f stripe=%s",
            tile_i, tile_j,
            static_cast<long>(acc32[0]),
            static_cast<double>(s_rf_val),
            static_cast<double>(acc_fp),
            static_cast<double>(activation_scale),
            static_cast<double>(contrib),
            static_cast<double>(stripe_out[0] + contrib),
            use_stripe_scales ? "1" : "0");
      }

      if (native_q8)
      {
        for (size_t i = 0; i < tile_i_actual; ++i)
        {
          float *row_out = stripe_out + i * out_row_stride;
          const double *acc_row = acc_fp.data() + i * tile_j_actual;
          float row_activation_scale = activation_scale;
          if (token_meta != nullptr && tile_i + i < token_meta->scales.size())
          {
            const float scale = token_meta->scales[tile_i + i];
            row_activation_scale = std::isfinite(scale) && scale > 0.0f ? scale : 1.0f;
          }
          for (size_t j = 0; j < tile_j_actual; ++j)
            row_out[j * out_col_stride] += static_cast<float>(acc_row[j] * static_cast<double>(row_activation_scale));
        }
      }
      else if (use_stripe_scales)
      {
        ggml::gemmini::update_q80_r_output_impl(
            output_stripe,
            acc32.data(),
            tile_j_actual,
            [s_rf_val = args->s_rf_stripe[stripe_col_idx]](size_t) { return s_rf_val; },
            activation_scale,
            stripe_out,
            out_row_stride,
            out_col_stride);
      }
      else
      {
        ggml::gemmini::update_q80_r_output_impl(
            output_stripe,
            acc32.data(),
            tile_j_actual,
            [s_rf = args->s_rf](size_t j) { return s_rf[j]; },
            activation_scale,
            stripe_out,
            out_row_stride,
            out_col_stride);
      }
    };

#if defined(GGML_GEMMINI_HAS_OPENMP)
    if (use_im2p_openmp)
    {
#pragma omp parallel for collapse(2) schedule(static) num_threads(im2p_threads)
      for (ptrdiff_t tile_i_idx = 0; tile_i_idx < static_cast<ptrdiff_t>(num_i_tiles); ++tile_i_idx)
      {
        for (ptrdiff_t tile_j_idx = 0; tile_j_idx < static_cast<ptrdiff_t>(num_j_tiles); ++tile_j_idx)
        {
          run_tile_pair(static_cast<size_t>(tile_i_idx), static_cast<size_t>(tile_j_idx));
        }
      }
    }
    else
#endif
    for (size_t tile_i_idx = 0; tile_i_idx < num_i_tiles; ++tile_i_idx)
    {
      for (size_t tile_j_idx = 0; tile_j_idx < num_j_tiles; ++tile_j_idx)
      {
        run_tile_pair(tile_i_idx, tile_j_idx);
      }
    }

    uint64_t end = ggml::gemmini::cycle::read();
    gemmini_log_cycle(layer, cycle_label, start, end);
  }

  static void tiled_matmul_im2p_sw(struct ggml_gemmini_args_t *args) {
    tiled_matmul_im2p_impl(args, true, "[tiled_matmul_auto_im2p] cpu.Q8_0_R tiled matmul");
  }

  // IM2P WS simulation. target system.
  // activation: ExSIA, weight: hierarchical block
  static void tiled_matmul_im2p_ws(struct ggml_gemmini_args_t *args) {
    tiled_matmul_im2p_impl(args, false, "[tiled_matmul_im2p_ws] ws-sim.Q8_0_R tiled matmul");
  }

  static void tiled_matmul_auto_im2p(struct ggml_gemmini_args_t *args) {
    if (args == NULL)
      return;

    const char *layer = ggml::gemmini::types::to_string(args->layer_type);
    const char *cpu_cycle_label = nullptr;
    const char *ws_cycle_label = nullptr;
    const char *os_message = nullptr;

    switch (args->weight_format)
    {
      case ggml_gemmini_args_t::im2p_weight_format_t::q8_h1:
        cpu_cycle_label = "[tiled_matmul_auto_im2p] cpu.Q8_H1 tiled matmul";
        ws_cycle_label = "[tiled_matmul_im2p_ws] ws-sim.Q8_H1 tiled matmul";
        os_message = "[tiled_matmul_auto_im2p] OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::im2p_weight_format_t::q8_h2:
        cpu_cycle_label = "[tiled_matmul_auto_im2p] cpu.Q8_H2 tiled matmul";
        ws_cycle_label = "[tiled_matmul_im2p_ws] ws-sim.Q8_H2 tiled matmul";
        os_message = "[tiled_matmul_auto_im2p] Q8_H2 OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1:
        cpu_cycle_label = "[tiled_matmul_auto_im2p] cpu.Q8_HP1 tiled matmul";
        ws_cycle_label = "[tiled_matmul_im2p_ws] ws-sim.Q8_HP1 tiled matmul";
        os_message = "[tiled_matmul_auto_im2p] Q8_HP1 OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2:
        cpu_cycle_label = "[tiled_matmul_auto_im2p] cpu.Q8_HP2 tiled matmul";
        ws_cycle_label = "[tiled_matmul_im2p_ws] ws-sim.Q8_HP2 tiled matmul";
        os_message = "[tiled_matmul_auto_im2p] Q8_HP2 OS mode is unsupported";
        break;
      case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel:
        if (!args->has_q8_channel_direct_read_contract())
          gemmini_log_debug_layer(layer,
              "[tiled_matmul_auto_im2p] reject malformed Q8_CHANNEL direct-read contract");
        else
          gemmini_log_debug_layer(layer,
              "[tiled_matmul_auto_im2p] Q8_CHANNEL direct-read requires baseline/DEC software route; IM2P hardware subroute would require repack");
        return;
      case ggml_gemmini_args_t::im2p_weight_format_t::q8_h0:
        gemmini_log_debug_layer(
            layer,
            "[tiled_matmul_auto_im2p] unsupported IM2P weight format=%d (not yet implemented in software path)",
            static_cast<int>(args->weight_format));
        return;
      default:
        gemmini_log_debug_layer(layer,
            "[tiled_matmul_auto_im2p] unsupported IM2P weight format=%d",
            static_cast<int>(args->weight_format));
        return;
    }

    if (args->tiled_matmul_type == CPU)
    {
      tiled_matmul_im2p_impl(args, true, cpu_cycle_label);
      return;
    }
    if (args->tiled_matmul_type == WS)
    {
      tiled_matmul_im2p_impl(args, false, ws_cycle_label);
      return;
    }

    gemmini_log_debug_layer(layer, os_message);
  }

  enum class baseline_activation_quant_t : uint8_t {
    FLOAT = 0,
    EXSIA,
    TENSOR,
    TOKEN,
    BLOCK,
  };

  enum class baseline_weight_quant_t : uint8_t {
    FLOAT = 0,
    TENSOR,
    CHANNEL,
    BLOCK,
  };

  struct baseline_route_t {
    baseline_activation_quant_t activation_quant;
    baseline_weight_quant_t weight_quant;
    const char *cpu_cycle_label;
    const char *npu_cycle_label;
  };

  static const baseline_route_t *baseline_route_for(
      baseline_activation_quant_t activation_quant,
      baseline_weight_quant_t weight_quant) {
    static const baseline_route_t routes[] = {
      {baseline_activation_quant_t::EXSIA, baseline_weight_quant_t::TENSOR,
       "[tiled_matmul_auto_A-exsia_B-tensor] cpu.dense_i8 tiled matmul",
       "[tiled_matmul_auto_A-exsia_B-tensor] npu.dense_i8 tiled matmul"},
      {baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::TENSOR,
       "[tiled_matmul_auto_A-tensor_B-tensor] cpu.dense_i8 tiled matmul",
       "[tiled_matmul_auto_A-tensor_B-tensor] npu.dense_i8 tiled matmul"},
      {baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::TENSOR,
       "[tiled_matmul_auto_A-token_B-tensor] cpu.dense_i8 tiled matmul",
       "[tiled_matmul_auto_A-token_B-tensor] npu.dense_i8 tiled matmul"},
      {baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::CHANNEL,
       "[tiled_matmul_auto_baseline] cpu.tensor_channel Q8_CHANNEL direct-read tiled matmul",
       "[tiled_matmul_auto_baseline] npu.tensor_channel Q8_CHANNEL direct-read tiled matmul"},
      {baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::BLOCK,
       "[tiled_matmul_auto_baseline] cpu.tensor_block dense_i8 tiled matmul",
       "[tiled_matmul_auto_baseline] npu.tensor_block dense_i8 tiled matmul"},
      {baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::CHANNEL,
       "[tiled_matmul_auto_baseline] cpu.token_channel Q8_CHANNEL direct-read tiled matmul",
       "[tiled_matmul_auto_baseline] npu.token_channel Q8_CHANNEL direct-read tiled matmul"},
      {baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::BLOCK,
       "[tiled_matmul_auto_baseline] cpu.token_block dense_i8 tiled matmul",
       "[tiled_matmul_auto_baseline] npu.token_block dense_i8 tiled matmul"},
      {baseline_activation_quant_t::BLOCK, baseline_weight_quant_t::TENSOR,
       "[tiled_matmul_auto_baseline] cpu.block_tensor dense_i8 tiled matmul",
       "[tiled_matmul_auto_baseline] npu.block_tensor dense_i8 tiled matmul"},
      {baseline_activation_quant_t::BLOCK, baseline_weight_quant_t::CHANNEL,
       "[tiled_matmul_auto_baseline] cpu.block_channel dense_i8 tiled matmul",
       "[tiled_matmul_auto_baseline] npu.block_channel dense_i8 tiled matmul"},
      {baseline_activation_quant_t::BLOCK, baseline_weight_quant_t::BLOCK,
       "[tiled_matmul_auto_baseline] cpu.block_block dense_i8 tiled matmul",
       "[tiled_matmul_auto_baseline] npu.block_block dense_i8 tiled matmul"},
    };

    for (const auto &route : routes)
      if (route.activation_quant == activation_quant && route.weight_quant == weight_quant)
        return &route;

    return nullptr;
  }

  static bool baseline_weight_scale_provider(const ggml_gemmini_args_t &args,
                                             baseline_weight_quant_t weight_quant,
                                             size_t channel,
                                             size_t k_block,
                                             float &scale) {
    switch (weight_quant) {
      case baseline_weight_quant_t::TENSOR:
        if (!args.weight_i8_scale_active || !std::isfinite(args.weight_scale))
          return false;
        scale = args.weight_scale;
        return true;
      case baseline_weight_quant_t::CHANNEL:
      {
        const bool q8_channel_direct_read = args.has_q8_channel_direct_read_contract();
        const bool q8_channel_dense_sidecar = args.has_q8_channel_dense_sidecar_contract();
        if (q8_channel_direct_read) {
          if (channel >= args.J)
            return false;
          const float candidate = args.q8_channel_scale(channel);
          if (!std::isfinite(candidate))
            return false;

          scale = candidate;
          return true;
        }

        if (!q8_channel_dense_sidecar || channel >= args.J)
            return false;

        const float candidate = args.weight_channel_scales[channel];
        if (!std::isfinite(candidate))
          return false;

        scale = candidate;
        return true;
      }
      case baseline_weight_quant_t::BLOCK:
      {
        const size_t block_size = args.block_size_k ? args.block_size_k : GGML_GEMMINI_BLOCK_SIZE;
        if (args.B_scales == nullptr || args.J == 0 || args.K == 0 ||
            args.blocks_J != args.J || args.blocks_K == 0 || block_size == 0 ||
            args.K > std::numeric_limits<size_t>::max() - (block_size - 1) ||
            args.blocks_K != (args.K + block_size - 1) / block_size ||
            channel >= args.blocks_J || k_block >= args.blocks_K ||
            channel > (std::numeric_limits<size_t>::max() - k_block) / args.blocks_K)
          return false;

        const size_t offset = channel * args.blocks_K + k_block;
        const float candidate = args.B_scales[offset];
        if (!std::isfinite(candidate))
          return false;

        scale = candidate;
        return true;
      }
      case baseline_weight_quant_t::FLOAT:
        return false;
    }

    return false;
  }

  static bool baseline_activation_scale_provider(
      const ggml_gemmini_args_t &args,
      baseline_activation_quant_t activation_quant,
      std::vector<float> &scales) {
    const auto &storage = args.act_quant.storage();
    switch (activation_quant) {
      case baseline_activation_quant_t::EXSIA:
        if (std::get_if<ggml::gemmini::quants::act::exsia::Meta>(&storage) == nullptr)
          return false;
        break;
      case baseline_activation_quant_t::TENSOR:
        if (std::get_if<ggml::gemmini::quants::act::tensor::Meta>(&storage) == nullptr)
          return false;
        break;
      case baseline_activation_quant_t::TOKEN:
      {
        const auto *meta = std::get_if<ggml::gemmini::quants::act::token::Meta>(&storage);
        GGML_ASSERT(meta != nullptr && "TOKEN baseline route requires token::Meta");
        GGML_ASSERT(meta->scales.size() == args.I &&
                    std::all_of(meta->scales.begin(), meta->scales.end(),
                                [](float scale) { return std::isfinite(scale) && scale > 0.0f; }) &&
                    "TOKEN baseline route requires one finite positive activation scale per row");
        break;
      }
      case baseline_activation_quant_t::BLOCK:
      {
        const auto *block_meta = std::get_if<ggml::gemmini::quants::act::block::Meta>(&storage);
        GGML_ASSERT((block_meta != nullptr ||
                     std::get_if<ggml::gemmini::quants::act::stripe::Meta>(&storage) != nullptr) &&
                    "BLOCK baseline route requires block::Meta or stripe::Meta");
        if (block_meta != nullptr) {
          GGML_ASSERT(block_meta->scales.size() == args.I &&
                      std::all_of(block_meta->scales.begin(), block_meta->scales.end(),
                                  [](float scale) { return std::isfinite(scale) && scale > 0.0f; }) &&
                      "BLOCK baseline route requires one finite positive activation scale per row");
        }
        break;
      }
      case baseline_activation_quant_t::FLOAT:
        return false;
    }

    scales = ggml::gemmini::quants::act::activation_scales(args, args.I);
    return scales.size() == args.I;
  }

  static void tiled_matmul_auto_baseline_dense(
      struct ggml_gemmini_args_t *args,
      const baseline_route_t &route) {
    if (args == nullptr || args->A == nullptr || args->f_out == nullptr)
      return;

    const bool native_q8 = args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1 ||
        args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
        args->weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    if (route.activation_quant == baseline_activation_quant_t::TENSOR &&
        route.weight_quant == baseline_weight_quant_t::TENSOR && native_q8) {
      tiled_matmul_im2p_impl(args, args->tiled_matmul_type == CPU,
          "[tiled_matmul_auto_A-tensor_B-native-q8] tiled matmul");
      return;
    }

    if (args->B == nullptr)
      return;

    const size_t dim_I = args->I;
    const size_t dim_J = args->J;
    const size_t dim_K = args->K;
    if (dim_I == 0 || dim_J == 0 || dim_K == 0 || args->sB == 0)
      return;

    if (route.activation_quant == baseline_activation_quant_t::EXSIA &&
        route.weight_quant == baseline_weight_quant_t::TENSOR &&
        (!args->weight_i8_scale_active || !std::isfinite(args->weight_scale)))
      return;

    const bool q8_channel_route = route.weight_quant == baseline_weight_quant_t::CHANNEL;
    const bool q8_channel_direct_read = q8_channel_route &&
        args->has_q8_channel_direct_read_contract();
    const bool q8_channel_dense_sidecar = q8_channel_route &&
        args->has_q8_channel_dense_sidecar_contract();
    if (q8_channel_route && !q8_channel_direct_read && !q8_channel_dense_sidecar)
      return;

    if (route.activation_quant == baseline_activation_quant_t::TOKEN) {
      if (q8_channel_route) {
        GGML_ASSERT(args->transpose_B &&
                    (q8_channel_direct_read || q8_channel_dense_sidecar) &&
                    "TOKEN Q8_CHANNEL route requires transpose_B and a valid channel contract");
      } else {
        GGML_ASSERT(args->transpose_B && args->sB == args->K &&
                    "TOKEN baseline route requires dense row-major I8 weights");
      }
      if (route.weight_quant == baseline_weight_quant_t::TENSOR) {
        GGML_ASSERT(args->weight_i8_scale_active &&
                    std::isfinite(args->weight_scale) &&
                    args->weight_scale > 0.0f &&
                    "TOKEN/TENSOR requires a finite positive scalar weight scale");
      }
    }

    if (route.weight_quant == baseline_weight_quant_t::TENSOR &&
        args->weight_i8_scale_active && !std::isfinite(args->weight_scale))
      return;

    GGML_ASSERT(args->act == NO_ACTIVATION);

    std::vector<float> activation_scales;
    if (!baseline_activation_scale_provider(*args, route.activation_quant, activation_scales))
      return;

    const size_t block_size = args->block_size_k ? args->block_size_k : GGML_GEMMINI_BLOCK_SIZE;
    if (block_size == 0 || dim_K > std::numeric_limits<size_t>::max() - (block_size - 1))
      return;
    const size_t weight_block_count = route.weight_quant == baseline_weight_quant_t::BLOCK ?
        (dim_K + block_size - 1) / block_size : 1;
    for (size_t j = 0; j < dim_J; ++j)
      for (size_t k_block = 0; k_block < weight_block_count; ++k_block) {
        float ignored_scale = 0.0f;
        if (route.weight_quant != baseline_weight_quant_t::TENSOR || args->weight_i8_scale_active) {
          if (!baseline_weight_scale_provider(*args, route.weight_quant, j, k_block, ignored_scale))
            return;
        }
      }

    if (args->tile_I == 0 || args->tile_J == 0 || args->tile_K == 0)
      gemmini_set_tile(args);

    const char *layer = ggml::gemmini::types::to_string(args->layer_type);
    const uint64_t start = ggml::gemmini::cycle::read();
    const elem_t *A = args->A;
    const elem_t *B = args->B;
    const size_t stride_A = args->sA ? args->sA : dim_K;
    const size_t row_stride_B = args->sB;
    const size_t out_row_stride = args->stride_f_out ? args->stride_f_out : dim_J;
    const size_t out_col_stride = args->col_stride_f_out ? args->col_stride_f_out : 1;
    const bool block_weight_route = route.weight_quant == baseline_weight_quant_t::BLOCK;
    std::vector<acc_t> acc32(dim_I * dim_J, 0);
    std::vector<double> block_acc;
    if (block_weight_route)
      block_acc.assign(dim_I * dim_J, 0.0);

    for (size_t i = 0; i < dim_I; ++i)
      for (size_t j = 0; j < dim_J; ++j)
        args->f_out[i * out_row_stride + j * out_col_stride] = 0.0f;

    const bool use_tiled_matmul = args->tiled_matmul_type != CPU && !block_weight_route;
    if (use_tiled_matmul) {
      tiled_matmul(dim_I, dim_J, dim_K,
          A, B, args->D, acc32.data(),
          stride_A, row_stride_B, args->sD, dim_J,
          1.0f, args->scale_B, args->scale_D,
          args->act, args->scale, args->bert_scale, args->repeating_bias,
          args->tile_I, args->tile_J, args->tile_K,
          args->transpose_A, args->transpose_B,
          true, args->low_D,
          args->weightA,
          args->tiled_matmul_type);
    } else {
      for (size_t i = 0; i < dim_I; ++i)
        for (size_t j = 0; j < dim_J; ++j) {
          int32_t acc = 0;
          for (size_t k = 0; k < dim_K; ++k) {
            const elem_t a = args->transpose_A ? A[k * stride_A + i] : A[i * stride_A + k];
            const int32_t product = static_cast<int32_t>(a) *
                static_cast<int32_t>(B[j * row_stride_B + k]);
            if (block_weight_route) {
              float weight_scale = 0.0f;
              if (!baseline_weight_scale_provider(*args, route.weight_quant, j, k / block_size, weight_scale))
                return;
              block_acc[i * dim_J + j] += static_cast<double>(product) * weight_scale;
            } else {
              acc += product;
            }
          }
          if (!block_weight_route)
            acc32[i * dim_J + j] = acc;
        }
    }

    if (route.weight_quant == baseline_weight_quant_t::TENSOR && !args->weight_i8_scale_active) {
      ggml::gemmini::dequantize(*args, 0, dim_K, acc32.data(), dim_J);
    } else {
      for (size_t i = 0; i < dim_I; ++i)
        for (size_t j = 0; j < dim_J; ++j) {
          const size_t output_offset = i * out_row_stride + j * out_col_stride;
          if (block_weight_route) {
            args->f_out[output_offset] += static_cast<float>(block_acc[i * dim_J + j] * activation_scales[i]);
          } else {
            float weight_scale = 0.0f;
            if (!baseline_weight_scale_provider(*args, route.weight_quant, j, 0, weight_scale))
              return;
            args->f_out[output_offset] += static_cast<float>(
                static_cast<double>(acc32[i * dim_J + j]) * weight_scale * activation_scales[i]);
          }
        }
    }

    const uint64_t end = ggml::gemmini::cycle::read();
    gemmini_log_cycle(layer,
        args->tiled_matmul_type == CPU ? route.cpu_cycle_label : route.npu_cycle_label,
        start, end);
  }

  static void tiled_matmul_auto_A_exsia_B_tensor(struct ggml_gemmini_args_t *args) {
    tiled_matmul_auto_baseline_dense(args, *baseline_route_for(
        baseline_activation_quant_t::EXSIA, baseline_weight_quant_t::TENSOR));
  }

  static void tiled_matmul_auto_A_tensor_B_tensor(struct ggml_gemmini_args_t *args) {
    tiled_matmul_auto_baseline_dense(args, *baseline_route_for(
        baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::TENSOR));
  }

  static void tiled_matmul_auto_A_token_B_tensor(struct ggml_gemmini_args_t *args) {
    tiled_matmul_auto_baseline_dense(args, *baseline_route_for(
        baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::TENSOR));
  }

  static void tiled_matmul_auto_baseline(struct ggml_gemmini_args_t *args,
                                         baseline_activation_quant_t activation_quant,
                                         baseline_weight_quant_t weight_quant) {
    if (activation_quant == baseline_activation_quant_t::FLOAT &&
        weight_quant == baseline_weight_quant_t::FLOAT) {
      tiled_matmul_auto_fp(args);
      return;
    }

    const baseline_route_t *route = baseline_route_for(activation_quant, weight_quant);
    GGML_ASSERT(route != nullptr && "unsupported Gemmini baseline quantization pair");
    tiled_matmul_auto_baseline_dense(args, *route);
  }

}} // namespace ggml::gemmini

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
//
// modified by Youngshin -- make gemmini_tiled_matmul_cycles to extern
#ifdef __cplusplus 
extern "C" {
#endif
extern volatile uint64_t gemmini_tiled_matmul_cycles; //added by Dongkyu
#ifdef __cplusplus
}
#endif

static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const bool double_buffered = tiled_matmul_type == WS;

    const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

    size_t tile_I, tile_J, tile_K;

    if (act == LAYERNORM || act == SOFTMAX) {
       tile_I = 1;
       tile_J = dim_J_padded/DIM;
       tile_K = 1;
    } else if (double_buffered) {
       tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
       tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
       tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
          (tile_J+1) * DIM <= dim_J_padded) {
        tile_J++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    exit(EXIT_SUCCESS);
#endif
#endif

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}


static void sp_tiled_conv(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int pool_out_row_dim, int pool_out_col_dim,

        int stride, int padding, int kernel_dim, int kernel_dilation,
        int in_stride, int weight_stride, int out_stride,

        int pool_size, int pool_stride, int pool_padding,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        const elem_t * input,
        const elem_t * weights,
        elem_t * output,
        const acc_t * bias,

        int act, acc_scale_t scale,

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        bool no_bias, bool no_pool, bool downsample, bool input_dilated,
        bool dw, int a_spad_id, int b_spad_id) {

  // When dw convs are true, we assume that kchs and ochs are 1
  if (dw) { kchs = 1; pochs = 1; }

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1)*(krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1)*(kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
  const int ichs = kchs;

#define UNDILATED(x) ((input_dilated) ? (((x)+1)/2) : (x))

  if (input_dilated) {
    irows_unpadded = (irows_unpadded+1)/2;
    icols_unpadded = (icols_unpadded+1)/2;

    irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
    icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
  }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  const bool transposed = trans_output_1203 || trans_input_3120 ||
      trans_weight_1203 || trans_weight_0132;
  int max_pixels_per_row = transposed || wrot180 || downsample ||
      input_dilated || kernel_dilation > 1 ||
      ichs > DIM ? 1 : DIM/ichs;
  if (max_pixels_per_row > kcols) max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  // Calculate spad address offsets
  const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  const int B_rows = trans_weight_0132 ?
    in_channels_per_bank * kcols * krows * ochs :
    out_channels_per_bank * kcols * krows * kchs;

  static uint32_t D_sp_addr_row = 0;
  static uint32_t C_sp_addr_row = 0;

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  if (bias != 0) {
    D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  if (output != 0) {
    C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  if (in_row_dim == in_col_dim && out_row_dim == out_col_dim && pool_out_row_dim == pool_out_col_dim) {
    gemmini_loop_conv_ws(batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, dw, a_spad_id, b_spad_id);
    return;
  }

  if (!no_pool) {
    printf("Pooling with rectangular convolutions is currently not supported.\n");
    exit(1);
  }

  // Only rectangular convolutions will use the following C code

  // mvin bias
  if (bias != NULL) {
    // TODO we probably don't need quite this many nested loops for this part

    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
        MAX_BLOCK_LEN_ACC * DIM;

    gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

    for (int b = 0; b < batches; b++)
      for (int orow = 0; orow < orows; orow++)
        for (int ocol = 0; ocol < ocols; ocol += DIM) {
          const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

          for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
            const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

            const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

            const acc_t * bias_dram_addr = no_bias ? NULL : bias + och;

            gemmini_extended_mvin3(bias_dram_addr,
                    D_sp_addr,
                    J, I);
          }
        }
  }

  // mvin input
  if (input != NULL){
    int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
      MAX_BLOCK_LEN * DIM;
    if (trans_input_3120) {
      max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
        MAX_BLOCK_LEN * DIM;
    }

    const int dram_stride = trans_input_3120 ?
      batch_size * sizeof(elem_t) :
      in_channels * sizeof(elem_t);

    const int spad_stride = trans_input_3120 ?
      ichs * (irows >> downsample) * (icols >> downsample) :
      batches * (irows >> downsample) * (icols >> downsample);

    gemmini_extended5_config_ld(dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, max_pixels_per_row, 0);

    const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
    const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

    for (int b = 0; b < batches; b += b_it)
      for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
        const int irow_padded = irow + UNDILATED(upad);

        for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
          // TODO There might be some unnecessary mvins here at the edge of the image

          int I = icols_unpadded - icol > (DIM << downsample) ?
            (DIM << downsample) : icols_unpadded - icol;

          if (icol < 0) {
            I = -icol > DIM ? DIM : -icol;
          } else if (icol >= icols_unpadded) {
            I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
          }

          const int icol_padded = icol + UNDILATED(lpad);

          for (int ich = 0; ich < ichs; ich += ich_it) {
            int K = ichs - ich > max_chs_per_mvin ?
              max_chs_per_mvin : ichs - ich;
            if (trans_input_3120) {
              K = batches - b > max_chs_per_mvin ?
                max_chs_per_mvin : batches - b;
            }

#define DS(x) ((x) >> (downsample))

            uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            if (trans_input_3120) {
              A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            }

            const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

            const elem_t * in = input + (b*in_row_dim*in_col_dim + irow*in_col_dim + icol) * in_stride + ich;
            if (is_zeros) {
              in = NULL;
            } else if (trans_input_3120) {
              in = input + (ich*in_row_dim*in_col_dim + irow*in_col_dim + icol) * batch_size + b;
            }

            gemmini_extended_mvin(in,
                A_sp_addr,
                K, I >> downsample);
          }

          icol += I;
        }
      }
  }

  // mvin weights
  if (weights != NULL) {
    int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
        MAX_BLOCK_LEN * DIM;
    if (trans_weight_0132) {
      max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
          MAX_BLOCK_LEN * DIM;
    }

    size_t dram_stride = weight_stride * sizeof(elem_t);
    if (dw) {
      dram_stride = sizeof(elem_t);
    } else if (trans_weight_1203) {
      dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
    } else if (trans_weight_0132) {
      dram_stride = in_channels * sizeof(elem_t);
    }

    const size_t spad_block_stride = trans_weight_0132 ?
      krows * kcols * ochs : krows * kcols * kchs;

    gemmini_extended4_config_ld(dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

    const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
    const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

    for (int och = 0; och < ochs; och += och_it) {
      for (int krow = 0; krow < krows; krow++)
        for (int kcol = 0; kcol < kcols; kcol++)
          for (int kch = 0; kch < kchs; kch += kch_it) {
            int K = kchs - kch > DIM ? DIM : kchs - kch;
            int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
            if (trans_weight_0132) {
              K = ochs - och > DIM ? DIM : ochs - och;
              J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
            }

            uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
            if (trans_weight_0132) {
              B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
            }

            const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * weight_stride + och;
            if (dw) {
              w = weights + krow * kernel_dim + kcol;
            } else if (trans_weight_1203) {
              w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
            } else if (trans_weight_0132) {
              w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
            }

            gemmini_extended_mvin2(w, B_sp_addr, J, K);
          }
    }
  }

  // Compute
  {
    const int b_it = trans_input_3120 ? DIM : 1;
    const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

    if (trans_input_3120) {
      gemmini_extended3_config_ex(0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, true);
    }

    for (int och = 0; och < ochs; och += DIM) {
      for (int krow = 0; krow < krows; krow++) {
        for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
          for (int kch = 0; kch < kchs; kch += DIM) {
            bool new_weights = true;

            for (int b = 0; b < batches; b += b_it) {
              for (int orow = 0; orow < orows; orow++) {
                // Skip some kernel rows due to input-dilation
                if (input_dilated && ((krow * kernel_dilation + orow * stride - upad) % 2 != 0)) {
                  continue;
                }

                for (int ocol = 0; ocol < ocols;) {
                  // Skip some cols dimensions due to input-dilation
                  if (input_dilated && ((kcol + ocol * stride - lpad) % 2 != 0)) {
                    ocol++;
                    continue;
                  }

                  int irow = orow * stride + krow * kernel_dilation;
                  int icol = ocol * stride + kcol * kernel_dilation;

                  if (input_dilated) {
                    irow = (irow + 1) / 2;
                    icol = (icol + 1) / 2;
                  }

                  const int pixels = kcols - kcol > max_pixels_per_row ?
                    max_pixels_per_row : kcols - kcol;

                  const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

                  // Over here, construct a new matrix
                  //
                  // Let us assume that we only ever operate on
                  // one pixel in one row.
                  // Thus, krows == kcols == 1
                  //
                  // Then, for every set of I, J, and K values
                  //     - I = ocols
                  //     - J = ochs
                  //     - K = kchs

                  int I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
                  const int J = ochs - och > DIM ? DIM : ochs - och;
                  const int K = pixels * (kchs - kch > DIM ? DIM : kchs - kch);

                  if (trans_input_3120) {
                    I = batches - b > DIM ? DIM : batches - b;
                  }

                  uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  if (trans_input_3120) {
                    A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  }

                  const int krow_ = wrot180 ? krows - krow - 1 : krow;
                  const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

                  uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
                  if (trans_weight_0132) {
                    B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
                  }

                  const uint32_t pre_sp_addr = new_weights ?
                    B_sp_addr : GARBAGE_ADDR;

                  // perform matmul
                  gemmini_extended_preload(pre_sp_addr, C_sp_addr, J, K, J, I);

                  if (new_weights) {
                    gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  } else {
                    gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  }

                  ocol += ocol_it;
                  new_weights = false;
                }
              }
            }
          }
        }
      }
    }
  }

#undef DS
#undef UNDILATED

  // mvout output
  if (output != NULL) {
    if (no_pool) {
      for (int b = 0; b < batches; b++)
        for (int orow = 0; orow < orows; orow++)
          for (int ocol = 0; ocol < ocols; ocol += DIM) {
            const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

            for (int och = 0; och < ochs; och += DIM) {
              const int J = ochs - och > DIM ? DIM : ochs - och;

              const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

              elem_t * out = output + (b*out_row_dim*out_col_dim + orow*out_col_dim + ocol) * out_stride + och;
              if (trans_output_1203) {
                out = output + (orow*out_col_dim*batch_size + ocol*batch_size + b) * out_channels + och;
              }

              gemmini_extended_mvout(out,
                  C_sp_addr,
                  J, I);
            }
          }
    } else {
      printf("Pooling with rectangular convolutions is currently not supported.\n");
      exit(1);

      /*
      gemmini_extended2_config_st(out_channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_row_dim, porows, pocols, orows, ocols, pupad, plpad);

      for (int b = 0; b < batches; b++) {
        for (int poch = 0; poch < pochs; poch += DIM) {
          const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

          elem_t * pout = output + (b * pool_out_row_dim * pool_out_col_dim)*out_channels + poch;

          const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

          gemmini_extended_mvout(pout,
              C_sp_addr,
              channels, 0);
        }
      }

      gemmini_extended_config_st(out_channels * sizeof(elem_t), act, scale);
      */
    }
  }
}


static int tiled_conv_total_spad_rows_dw(bool acc, bool weight,
        int stride,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int irows = orows * stride + krows - 1; // - 2 * padding;
    const int icols = ocols * stride + kcols - 1; // - 2 * padding;
    const int ichs = kchs;

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);

    const int A_rows = in_channels_per_bank * batches * irows * icols;
    const int B_rows = out_channels_per_bank * kcols * krows * kchs;
    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    if (acc)
        return C_rows;
    else if(weight)
        return B_rows;
    else
        return A_rows;
}


static int tiled_conv_total_spad_rows(bool acc,
        int stride,
        int input_dilation,
        int kernel_dilation,
        bool downsample,
        bool trans_weight_0132,
        bool trans_input_3120,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
    const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

    int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
    int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
    const int ichs = kchs;

    irows = irows / input_dilation + (irows % input_dilation != 0);
    icols = icols / input_dilation + (icols % input_dilation != 0);

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int batches_per_bank = batches / DIM + (batches % DIM != 0);

    const int A_rows = trans_input_3120 ?
        (batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
        (in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

    const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    return acc ? C_rows : A_rows + B_rows;
}


static void conv_cpu_without_pool(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_row_dim; orow++) {
      for (int ocol = 0; ocol < out_col_dim; ocol++) {
        for (int och = 0; och < out_channels; och++) {

          acc_t opixel = no_bias ? 0 : bias[och];

          for (int krow = 0; krow < kernel_dim; krow++) {
            if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
              continue;

            const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                continue;

              const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

              for (int kch = 0; kch < in_channels; kch++) {
                const elem_t *in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_stride + kch;
                if (trans_input_3120) {
                  // NHWC to CHWN
                  in = input + (kch * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b;
                }

                elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                    0 : *in;

                const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * weight_stride + och);
                if (trans_weight_1203) {
                  // HWIO to WIHO
                  weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + och);
                } else if (trans_weight_0132) {
                  // HWIO to HWOI
                  weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + och) * in_channels + kch);
                }

                opixel += weight * ipixel;
              }
            }
          }

          elem_t *out = output + (b * out_row_dim * out_col_dim + orow * out_col_dim + ocol) * out_stride + och;
          if (trans_output_1203) {
            // NHWC to HWNC
            out = output + (orow * out_col_dim * batch_size + ocol * batch_size + b) * out_channels + och;
          }

          *out = scale_and_sat(opixel, act, scale, 0);
        }
      }
    }
  }
}


static void conv_dw_cpu_without_pool(
        int batch_size, int in_row_dim, int in_col_dim,
        int channels, int out_row_dim, int out_col_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_row_dim; orow++) {
      for (int ocol = 0; ocol < out_col_dim; ocol++) {
        for (int ch = 0; ch < channels; ch++) {
          acc_t opixel = no_bias ? 0 : bias[ch];

          for (int krow = 0; krow < kernel_dim; krow++) {
            const int irow = orow * stride + krow - padding;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              const int icol = ocol * stride + kcol - padding;

              const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * channels + ch;

              const elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                  0 : *in;

              const elem_t weight = *(weights + (ch * kernel_dim + krow) * kernel_dim  + kcol);

              opixel += weight * ipixel;
            }
          }

          elem_t *out = output + (b * out_row_dim * out_col_dim + orow * out_col_dim + ocol) * channels + ch;

          *out = scale_and_sat(opixel, act, scale, 0);
        }
      }
    }
  }
}


static void conv_cpu(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_cpu_without_pool(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_row_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_col_dim; pocol++) {
        for (int poch = 0; poch < out_channels; poch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_row_dim || ocol < 0 || ocol >= out_col_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[poch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
                    continue;

                  const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                      continue;

                    const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

                    for (int kch = 0; kch < in_channels; kch++) {
                      const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_stride + kch;
                      if (trans_input_3120) {
                        // NHWC to CHWN
                        in = input + (kch * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b;
                      }

                      elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                          0 : *in;

                      const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                      const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                      elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * weight_stride + poch);
                      if (trans_weight_1203) {
                        // HWIO to WIHO
                        weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + poch);
                      } else if (trans_weight_0132) {
                        // HWIO to HWOI
                        weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch);
                      }

                      opixel += weight * ipixel;
                    }
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, 0);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_stride + poch;
                if (trans_output_1203) {
                  // NHWC to HWNC
                  out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                }

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void conv_dw_cpu(
        int batch_size, int in_row_dim, int in_col_dim,
        int channels, int out_row_dim, int out_col_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_dw_cpu_without_pool(
        batch_size, in_row_dim, in_col_dim,
        channels, out_row_dim, out_col_dim,
        stride, padding, kernel_dim,
        input, weights, bias, output,
        act, scale);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_row_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_col_dim; pocol++) {
        for (int ch = 0; ch < channels; ch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_row_dim || ocol < 0 || ocol >= out_col_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[ch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  const int irow = orow * stride + krow - padding;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    const int icol = ocol * stride + kcol - padding;

                    const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * channels + ch;

                    elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                        0 : *in;

                    const elem_t weight = *(weights + (ch * kernel_dim + krow) * kernel_dim  + kcol);

                    opixel += weight * ipixel;
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, 0);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * channels + ch;

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void tiled_conv(
        int batch_size,
        int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132) {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      // assume in_dim_rows = in_dim_cols
      // and out_dim_rows = out_dim_cols for now
      conv_cpu(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const bool downsample = stride == 2 && kernel_dim == 1 && in_row_dim % 2 == 0 && in_col_dim % 2 == 0
      && padding == 0 && no_pool && input_dilation == 1 && !trans_input_3120;

    const int input_dilated = input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
        if (input_dilation > 2) {
            printf("input_dilation > 2 is only supported on CPU\n");
            exit(1);
        }
        if (input_dilation > 1 && stride > 1) {
            printf("input input_dilation is only supported when stride == 1\n");
            exit(1);
        }
        if (trans_output_1203 && !no_pool) {
            printf("Output can only be transposed when pooling is disabled\n");
            exit(1);
        }
        if (trans_input_3120 && trans_weight_0132) {
            printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = trans_output_1203 ?
        batch_size * out_channels * sizeof(elem_t) :
        out_stride * sizeof(elem_t);
    gemmini_extended_config_st(st_dram_stride, act, scale);

    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_row_dim = in_row_dim + (input_dilation - 1) * (in_row_dim- 1);
    const int dilated_in_col_dim = in_col_dim + (input_dilation - 1) * (in_col_dim- 1);

    size_t a_spad_id = 0;
    size_t b_spad_id = 0;

    int porow_end = pool_out_row_dim;
	int porow_start = 0;
    bool a_reuse = false;
    bool b_reuse = false;
    size_t num_kch = ceil_divide_int(in_channels, kchs);
    size_t num_poch = ceil_divide_int(out_channels, pochs);
    size_t num_b = ceil_divide_int(batch_size, batches);
    size_t num_porow = ceil_divide_int((porow_end - porow_start), porows);
    size_t num_pocol = ceil_divide_int(pool_out_col_dim, pocols);
    size_t num_krow = ceil_divide_int(kernel_dim, krows);
    size_t num_kcol = ceil_divide_int(kernel_dim, kcols);


//    printf("num_kch: %d, num_poch: %d, num_b: %d, num_porow: %d, num_pocol: %d, num_krow: %d, num_kcol: %d\n", num_kch, num_poch, num_b, num_porow, num_pocol, num_krow, num_kcol);

    if(num_kch * num_poch * num_krow * num_kcol <= 2) 
      b_reuse = true;
    if(num_kch * num_krow * num_kcol * num_b * num_porow * num_pocol <= 2)
      a_reuse = true;

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = porow_start; porow < porow_end; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int poch = 0; poch < out_channels; poch += pochs) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow * kernel_dilation - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

                            for (int kch = 0; kch < in_channels; kch += kchs) {
                                if(a_reuse)
						           a_spad_id = (kch + krow + kcol + b + (porow - porow_start) + pocol) == 0 ? 1 : 2;
					            if(b_reuse)
						           b_spad_id = (kch + poch + krow + kcol) == 0 ? 1 : 2;
                                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_stride + poch;
                                if (trans_output_1203) {
                                    out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                                }

                                if (krow + krows < kernel_dim ||
                                        kcol + kcols < kernel_dim ||
                                        kch + kchs < in_channels) {
                                    out = NULL;
                                }

                                const acc_t * bias_ = bias + poch;
                                if (krow > 0 ||
                                        kcol > 0 ||
                                        kch > 0) {
                                    bias_ = NULL;
                                }

                                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                                const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                                const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                                const int orows_ = porows_ * pool_stride + pool_size - 1;

                                const int plpad = ocol < 0 ? -ocol : 0;
                                const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                                const int pupad = orow < 0 ? -orow : 0;
                                const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

                                const int dilated_krows_ = krows_ + (kernel_dilation - 1)*(krows_ - 1);
                                const int dilated_kcols_ = kcols_ + (kernel_dilation - 1)*(kcols_ - 1);

                                const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                                const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                                int lpad = icol < 0 ? -icol : 0;
                                int rpad = icol + icols_ > dilated_in_col_dim ? icol + icols_ - dilated_in_col_dim : 0;
                                int upad = irow < 0 ? -irow : 0;
                                int dpad = irow + irows_ > dilated_in_row_dim ? irow + irows_ - dilated_in_row_dim : 0;

                                if (input_dilated) {
                                  lpad += lpad == 0 && icol % 2 != 0;
                                  rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                                  upad += upad == 0 && irow % 2 != 0;
                                  dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                                }

                                int krow_ = krow;
                                int kcol_ = kcol;
                                if (wrot180) {
                                  krow_ = kernel_dim - krow - krows_;
                                  kcol_ = kernel_dim - kcol - kcols_;
                                }

                                const elem_t * weights_slice = weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * weight_stride + poch;
                                if (trans_weight_1203) {
                                  weights_slice = weights + (kch*kernel_dim*kernel_dim + krow_*kernel_dim+kcol_) * out_channels + poch;
                                } else if (trans_weight_0132) {
                                  weights_slice = weights + (krow_*kernel_dim*out_channels + kcol_*out_channels + poch) * in_channels + kch;
                                }

                                const elem_t * in = input + (b *in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * in_stride + kch;
                                if (trans_input_3120) {
                                  in = input + (kch * in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * batch_size + b;
                                }
                                if(b_reuse && (pocol + (porow - porow_start) + b > 0)) weights_slice = NULL;
							    if(a_reuse && (poch > 0)) in = NULL;
                                //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, in: %llu, weight: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, in, weights_slice);
 
                                sp_tiled_conv(
                                    batch_size, in_row_dim, in_col_dim, in_channels,
                                    out_channels, out_row_dim, out_col_dim,
                                    pool_out_row_dim, pool_out_col_dim,

                                    stride, padding, kernel_dim, kernel_dilation,
                                    in_stride, weight_stride, out_stride,

                                    pool_size, pool_stride, pool_padding,

                                    batches_,
                                    porows_, pocols_, pochs_,
                                    krows_, kcols_, kchs_,

                                    lpad, rpad, upad, dpad,
                                    plpad, prpad, pupad, pdpad,

                                    in,
                                    weights_slice,
                                    out,
                                    bias_,

                                    act, scale,

                                    wrot180, trans_output_1203, trans_input_3120,
                                    trans_weight_1203, trans_weight_0132,

                                    no_bias, no_pool, downsample, input_dilated,
                                    false, a_spad_id, b_spad_id);

                            }
                        }
                    }
                }
            }
        }
    }
}


static void tiled_conv_dw(
    int batch_size, int in_row_dim, int in_col_dim,
    int channels, int out_row_dim, int out_col_dim,
    int stride, int padding, int kernel_dim,

    int batches,
    int porows, int pocols,
    int krows, int kcols,

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      conv_dw_cpu(
        batch_size, in_row_dim, in_col_dim,
        channels, out_row_dim, out_col_dim,
        stride, padding, kernel_dim,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            batches, porows, pocols, 1, krows, kcols, 1, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            batches, porows, pocols, 1, krows, kcols, 1, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = channels * sizeof(elem_t);
    gemmini_extended_config_st(st_dram_stride, act, scale);

    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, stride, false, false, false);

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = 0; porow < pool_out_row_dim; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int ch = 0; ch < channels; ch++) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol - padding;

                            elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * channels + ch;

                            if (krow + krows < kernel_dim ||
                                    kcol + kcols < kernel_dim) {
                                out = NULL;
                            }

                            const acc_t * bias_ = bias + ch;
                            if (krow > 0 ||
                                    kcol > 0) {
                                bias_ = NULL;
                            }

                            const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                            const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                            const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                            const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                            const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;

                            const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                            const int orows_ = porows_ * pool_stride + pool_size - 1;

                            const int plpad = ocol < 0 ? -ocol : 0;
                            const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                            const int pupad = orow < 0 ? -orow : 0;
                            const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

                            const int icols_ = (ocols_ - plpad - prpad) * stride + kcols_ - 1;
                            const int irows_ = (orows_ - pupad - pdpad) * stride + krows_ - 1;

                            int lpad = icol < 0 ? -icol : 0;
                            int rpad = icol + icols_ > in_col_dim ? icol + icols_ - in_col_dim : 0;
                            int upad = irow < 0 ? -irow : 0;
                            int dpad = irow + irows_ > in_row_dim ? irow + irows_ - in_row_dim : 0;

                            const elem_t * weights_slice = weights + (ch*kernel_dim + krow) * kernel_dim + kcol;

                            const elem_t *in = input + (b * in_row_dim * in_col_dim + (irow+upad) * in_col_dim + (icol+lpad)) * channels + ch;

                            sp_tiled_conv(
                                batch_size, in_row_dim, in_col_dim, channels,
                                channels, out_row_dim, out_col_dim,
                                pool_out_row_dim, pool_out_col_dim,

                                stride, padding, kernel_dim, 1,
                                channels, 1, channels,

                                pool_size, pool_stride, pool_padding,

                                batches_,
                                porows_, pocols_, 1,
                                krows_, kcols_, 1,

                                lpad, rpad, upad, dpad,
                                plpad, prpad, pupad, pdpad,

                                in,
                                weights_slice,
                                out,
                                bias_,

                                act, scale,

                                false, false, false,
                                false, false,

                                no_bias, no_pool, false, false,
                                true, 0, 0);

                        }
                    }
                }
            }
        }
    }
}

// need to specify each operand/output's stride
// stride only for trans == false, wrot == false
static void tiled_conv_stride_auto(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride, // specify in/output's stride
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};
    const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;
    const int in_channels_idx = 6;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows(false,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows(true,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];

    /*
    spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    */

#ifdef PRINT_TILE
#if PRINT_TILE
    printf("batches = %d\n", batches);
    printf("orows   = %d\n", orows);
    printf("ocols   = %d\n", ocols);
    printf("ochs    = %d\n", ochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);

    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
#endif
#endif

    tiled_conv(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        input,
        weights,
        bias,
        output,

        act, scale,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);
}


static void tiled_conv_auto(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim, 
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

    int in_stride = in_channels;
    int out_stride = out_channels;
    int weight_stride = out_channels;
    tiled_conv_stride_auto(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        
        input, weights, bias, output,

        act, scale, pool_size, pool_stride, pool_padding,
        tiled_conv_type);

}

// This function is for a convolution with kernel_dim=1, stride==2, padding=0, and no pooling
static void tiled_conv_downsample(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int in_stride, int weight_stride, int out_stride,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,

        enum tiled_matmul_type_t tiled_conv_type) {

    // Rectangular dimensions for this function are currently not supported
    if (in_row_dim != in_col_dim || out_row_dim != out_col_dim) {
        printf("Rectangular convolutions for tiled_conv_downsample are currently not supported.\n");
        exit(1);
    }

    const int in_dim = in_row_dim;
    const int out_dim = out_row_dim;

    const int stride = 2;

    for (int b = 0; b < batch_size; b++) {
        for (int irow = 0; irow < in_row_dim; irow += stride) {
            const int orow = irow / stride;

            const int I = in_col_dim / stride; // number of columns in row
            const int J = out_channels;
            const int K = in_channels;

            const elem_t * A = input + (b * in_dim + irow) * in_dim * in_stride;
            const elem_t * B = weights;
            const acc_t * D = bias;
            elem_t * C = output + (b * out_dim + orow) * out_dim * out_stride;

            const int A_stride = in_stride * 2;
            const int B_stride = weight_stride;
            const int D_stride = out_stride;
            const int C_stride = out_stride;

            tiled_matmul_auto(I, J, K, A, B, (void*)D, (void*)C,
                    A_stride, B_stride, D_stride, C_stride,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, act, scale, 0,
                    true, false, false, false, false, 0, tiled_conv_type);
        }
    }
}

//for mobilenet's depthwise convs
static void tiled_conv_dw_auto(
    int batch_size, int in_row_dim, int in_col_dim,
    int channels, int out_row_dim, int out_col_dim,
    int stride, int padding, int kernel_dim,

    elem_t * input,
    elem_t * weights,
    acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, 1, kernel_dim, kernel_dim, 1};
    const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, 1, kernel_dim, kernel_dim, 1};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows(false,
                stride, 1, 1, false, false, false,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows(true,
                stride, 1, 1, false, false, false,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = 1; // args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = 1; // args[6];

    /*
    spad_rows = tiled_conv_total_spad_rows(false,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    printf("batches = %d\n", batches);
    printf("orows   = %d\n", orows);
    printf("ocols   = %d\n", ocols);
    printf("ochs    = %d\n", ochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);

    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
    */

    tiled_conv_dw(
        batch_size, in_row_dim, in_col_dim,
        channels, out_row_dim, out_col_dim,
        stride, padding, kernel_dim,

        batches,
        orows, ocols,
        krows, kcols,

        input,
        weights,
        bias,
        output,

        act, scale,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);
}


static void resadd_cpu(const size_t I, const size_t J,
        const size_t stride,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * stride + j;
            const elem_t * b = B + i * stride + j;
            elem_t * c = C + i * stride + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}


static void sp_tiled_resadd(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride,
        bool relu) {

    int pad_I = ((I%DIM) == 0) ? 0 : DIM - (I % DIM);
    int pad_J = ((J%DIM) == 0) ? 0 : DIM - (J % DIM);
    int tile_I = (I%DIM == 0) ? (int)(I/DIM) : (int)(I/DIM) + 1;
    int tile_J = (J%DIM == 0) ? (int)(J/DIM) : (int)(J/DIM) + 1;
    //printf("pad I: %d, pad_J: %d, tile_I: %d, tile_J: %d\n", pad_I, pad_J, tile_I, tile_J);
    gemmini_loop_ws(tile_I, tile_J, 0, pad_I, pad_J, 0, A, B, NULL, C, A_row_stride, B_row_stride, 0, C_row_stride, false, false, false, false, false, relu, 0, 0, true);
    /*
    // Use the new mvin2 command to overlap mvin A, mvin B, and mvout C

    size_t blocks = (J/DIM + (J % DIM != 0));
    if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

    const size_t rounded_up_J = (J / DIM + (J % DIM != 0)) * DIM;

    // Mvin A
    // printf("Mving A\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const A_dram_addr = A + i * A_row_stride + j;
            const uint32_t A_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;

            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
    }

    // Mvin B
     printf("Mving B\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const B_dram_addr = B + i * B_row_stride + j;
            const uint32_t B_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
        }
    }

    // Mvout C from accumulator
    // printf("Mvout C from accumulator\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            elem_t * const C_dram_addr = C + i * C_row_stride + j;
            const uint32_t C_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, cols, rows);
        }
    }
    */
}

// Compute MVIN_SCALE(A, A_scale) + MVIN_SCALE(B, B_scale) = C
static void tiled_resadd(const size_t I, const size_t J,
        const size_t stride,
        const size_t tile_I, const size_t tile_J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    gemmini_extended_config_st(stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
    gemmini_config_ex(WS, 0, 0);

    gemmini_extended4_config_ld(stride * sizeof(elem_t), A_scale, true, DIM, 0);
    gemmini_extended4_config_ld(stride * sizeof(elem_t), B_scale, true, DIM, 1);

    for (size_t i = 0; i < I; i += tile_I) {
        for (size_t j = 0; j < J; j += tile_J) {
            const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
            const size_t J_tile = j + tile_J <= J ? tile_J : J - j;

            const elem_t * a = A + i * stride + j;
            const elem_t * b = B + i * stride + j;
            elem_t * c = C + i * stride + j;

            sp_tiled_resadd(I_tile, J_tile,
                    A_scale, B_scale, a, b, c,
                    stride, stride, stride,
                    relu);
        }
    }

    gemmini_fence();
}

// Compute (A >> A_shift) + B = C
// specify stride
static void tiled_resadd_stride_auto(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const size_t stride,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    if (matadd_type == CPU) {
        resadd_cpu(I, J, stride,
            A_scale, B_scale, C_scale, A, B, C,
            relu);
        return;
    }

    size_t tile_I = I, tile_J = J;

    // size_t total_spad_rows = 2 * (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

    // TODO this is a very inefficient way of doing this...
    while (total_acc_rows > ACC_ROWS / 2) {
        //if(tile_J > MAX_BLOCK_LEN * DIM)
        //    tile_J = MAX_BLOCK_LEN * DIM;
        //else 
        if (tile_I >= tile_J || tile_J <= DIM)
            tile_I /= 2;
        else
            tile_J -= DIM;

        total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    }

    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);

    if (matadd_type == WS) {
      tiled_resadd(I, J, stride, tile_I, tile_J,
            A_scale, B_scale, C_scale, A, B, C,
            relu, matadd_type);
    }
    else {
      printf("Unsupported type\n");
      exit(1);
    }
}

static void tiled_resadd_auto(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {
    tiled_resadd_stride_auto(I, J, 
        A_scale, B_scale, C_scale,
        J,
        A, B, C,
        relu, matadd_type);
}

static void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum += input[pixel * channels + channel];
        }
      }

#ifdef ELEM_T_IS_FLOAT
      output[batch * channels + channel] = sum / count;
#else
      output[batch * channels + channel] = (sum + count/2) / count;
#endif
    }
  }
}


static void sp_tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim, int channel_tile_size) {
  const uint32_t C_acc_addr_start = ((uint32_t)1 << 31);

  size_t blocks = channel_tile_size/DIM + (channel_tile_size % DIM != 0);
  if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

  for (int channel = 0; channel < channel_tile_size; channel += blocks*DIM) {
    for (int row = 0; row < dim; row++) {
      for (int col = 0; col < dim; col++) {
        const elem_t * in = input +
          (row * dim + col) * channels +
          channel;

        const uint32_t acc_addr_start = C_acc_addr_start |
          ((row != 0 || col != 0) << 30);

        const uint32_t acc_addr = acc_addr_start + channel / DIM;

        const size_t cols = channel + blocks*DIM <= channel_tile_size ?
          blocks*DIM : channel_tile_size - channel;

        const size_t rows = 1;

        gemmini_extended_mvin(in, acc_addr, cols, rows);
      }
    }
  }

  for (int channel = 0; channel < channel_tile_size; channel += DIM) {
    elem_t * out = output + channel;

    const uint32_t acc_addr = C_acc_addr_start + channel / DIM;

    const size_t cols = channel + DIM <= channel_tile_size ?
      DIM : channel_tile_size - channel;

    const size_t rows = 1; // TODO we should move out more than just one row here

    gemmini_extended_mvout(out, acc_addr, cols, rows);
  }
}


static void tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    int channel_tile_size) {

  gemmini_extended4_config_ld(DIM*sizeof(elem_t), MVIN_SCALE_IDENTITY, true, 1, 0);
  gemmini_config_ex(0, NO_ACTIVATION, 0);
  gemmini_extended_config_st(0, NO_ACTIVATION, 1.0 / (dim*dim));

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel += channel_tile_size) {
      const int tile_size = channel + channel_tile_size <= channels ?
        channel_tile_size : channels - channel;

      sp_tiled_global_average(input + batch * dim * dim * channels + channel,
          output + batch * channels + channel,
          batches, channels, dim, tile_size);
    }
  }
}


static void tiled_global_average_auto(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type) {
  if (type == CPU) {
    return global_average_cpu(input, output, batches, channels, dim);
  }

  int channel_tile_size = channels;

  int acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  while (acc_rows > ACC_ROWS) {
    channel_tile_size--;
    acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  }

  tiled_global_average(input, output, batches, channels, dim,
      channel_tile_size);
}

#undef abs

#endif // SRC_MAIN_C_GEMMINI_H
