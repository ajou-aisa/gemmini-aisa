
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocc-software/src/xcustom.h"

#define DIM 64
#define GEN 1
#define SLOT 0

// First test: 0
// Second test: change to 1
#define MIXED_SCALES 1

#define ZERO_CARRIER 0x80000000U

static int8_t A[DIM][DIM]
    __attribute__((aligned(64)));

static int8_t B[DIM][DIM]
    __attribute__((aligned(64)));

static volatile int32_t C[DIM][DIM]
    __attribute__((aligned(64)));

static uint32_t carrier_for_column(int col) {
    if (!MIXED_SCALES)
        return 0;

    switch (col % 4) {
        case 0: return 0;            // x1
        case 1: return 1;            // x2
        case 2: return ZERO_CARRIER; // x0
        default: return 2;           // x4
    }
}

static uint64_t scale_key(int col) {
    return ((uint64_t)col << 16) |
           ((uint64_t)GEN << 8) |
           SLOT;
}

static uint64_t scale_write(int col, uint32_t carrier) {
    uint64_t status;
    ROCC_INSTRUCTION_R_R_R(
        3, status, scale_key(col), (uint64_t)carrier, 23
    );
    return status;
}

static uint64_t scale_release(int col) {
    uint64_t status;
    ROCC_INSTRUCTION_R_R_R(
        3, status, scale_key(col), 0ULL, 24
    );
    return status;
}

static uint64_t potal_fence(void) {
    uint64_t status;
    ROCC_INSTRUCTION_R_R_R(
        3, status, 0ULL, 0ULL, 25
    );
    return status;
}

static void configure_gemmini(void) {
    // CONFIG_LOAD: state 0 for A, state 1 for B.
    for (int id = 0; id < 2; ++id) {
        uint64_t rs1 =
            ((uint64_t)DIM << 16) |
            (1ULL << 8) |
            ((uint64_t)id << 3) |
            1ULL;

        ROCC_INSTRUCTION_0_R_R(3, rs1, DIM, 0);
    }

    // CONFIG_EX:
    // WS, no activation, shift 0,
    // A stride = 1, C stride = 1.
    uint64_t ex_rs1 = (1ULL << 16) | (1ULL << 2);
    uint64_t ex_rs2 = 1ULL << 48;

    ROCC_INSTRUCTION_0_R_R(3, ex_rs1, ex_rs2, 0);

    // CONFIG_STORE: 16 INT32 values per row.
    ROCC_INSTRUCTION_0_R_R(
        3, 2ULL, DIM * sizeof(int32_t), 0
    );
}

static void issue_potal_gemm(void) {
    // LOOP_WS_CONFIG_BOUNDS
    // I = J = K = 1 tile, no padding.
    uint64_t bounds =
        (1ULL << 32) |
        (1ULL << 16) |
        1ULL;

    ROCC_INSTRUCTION_0_R_R(3, 0ULL, bounds, 9);

    // A and B DRAM addresses.
    ROCC_INSTRUCTION_0_R_R(
        3,
        (uint64_t)(uintptr_t)&A[0][0],
        (uint64_t)(uintptr_t)&B[0][0],
        10
    );

    // No bias, C output.
    ROCC_INSTRUCTION_0_R_R(
        3,
        0ULL,
        (uint64_t)(uintptr_t)&C[0][0],
        11
    );

    // A/B tile strides.
    ROCC_INSTRUCTION_0_R_R(3, DIM, DIM, 12);

    // D/C tile strides.
    ROCC_INSTRUCTION_0_R_R(3, DIM, DIM, 13);

    // LOOP_WS configuration:
    // bit 1      : full_C = 1
    // bit 20     : PoTal enable
    // bits 36:21 : scale base = SLOT
    // bits 44:37 : generation = GEN
    // bits 60:45 : passesPerBlock = 1
    uint64_t loop_rs1 =
        (1ULL << 1) |
        (1ULL << 20) |
        ((uint64_t)SLOT << 21) |
        ((uint64_t)GEN << 37) |
        (1ULL << 45);

    // No transpose or resadd.
    ROCC_INSTRUCTION_0_R_R(3, loop_rs1, 0ULL, 8);
}

static int32_t cpu_reference(int row, int col) {
    int32_t sum = 0;

    for (int k = 0; k < DIM; ++k)
        sum += (int32_t)A[row][k] *
               (int32_t)B[k][col];

    uint32_t carrier = carrier_for_column(col);

    if (carrier == ZERO_CARRIER)
        return 0;

    return sum * (1 << carrier);
}

int main(void) {
    printf("POTAL_GEMM_D64 starting\n");

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            A[i][j] = (int8_t)((i * 7 + j * 5) % 13) - 6;
            B[i][j] = (int8_t)(
                ((i * 13 + j * 7) % 31) - 15
            );
            C[i][j] = -9999;
        }
    }

    asm volatile("fence rw,rw" ::: "memory");

    configure_gemmini();

    for (int col = 0; col < DIM; ++col) {
        uint64_t status =
            scale_write(col, carrier_for_column(col));

        if (status != 0) {
            printf("FAIL scale write col=%d status=%d\n",
                   col, (int)status);
            exit(1);
        }
    }

    printf("PASS scale initialization\n");

    issue_potal_gemm();
    printf("LOOP_WS issued; waiting for POTAL_FENCE\n");

    if (potal_fence() != 0) {
        printf("FAIL POTAL_FENCE\n");
        exit(1);
    }

    printf("PASS GEMM fence\n");
    asm volatile("fence rw,rw" ::: "memory");

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            int32_t expected = cpu_reference(i, j);
            int32_t actual = C[i][j];

            if (actual != expected) {
                printf(
                    "FAIL C[%d][%d]: expected=%d actual=%d\n",
                    i, j, expected, actual
                );
                exit(1);
            }
        }
    }

    printf("PASS GEMM numerical comparison\n");

    for (int col = 0; col < DIM; ++col) {
        if (scale_release(col) != 0) {
            printf("FAIL scale release col=%d\n", col);
            exit(1);
        }
    }

    printf("POTAL_GEMM_D64 PASS\n");
    exit(0);
}

