
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocc-software/src/xcustom.h"

#define DIM 32
#define K_TILES 2
#define KDIM (DIM * K_TILES)

#define SCALE_BASE 8
#define SCALE_GEN 7
#define ZERO_CARRIER 0x80000000U

// Case A: 2 (shared K32 scale)
// Case B: 1 (independent K16 scales)
#define PASSES_PER_BLOCK 1

#if K_TILES % PASSES_PER_BLOCK != 0
#error Invalid scale block configuration
#endif

#define SCALE_BLOCKS (K_TILES / PASSES_PER_BLOCK)

static int8_t A[DIM][KDIM]
    __attribute__((aligned(64)));

static int8_t B[KDIM][DIM]
    __attribute__((aligned(64)));

static volatile int32_t C[DIM][DIM]
    __attribute__((aligned(64)));

static uint32_t carrier_for(int block, int col) {
    switch ((col + block) % 4) {
        case 0: return 0;            // x1
        case 1: return 1;            // x2
        case 2: return ZERO_CARRIER; // x0
        default: return 2;           // x4
    }
}

static uint64_t key(int block, int col) {
    return ((uint64_t)col << 16) |
           ((uint64_t)SCALE_GEN << 8) |
           (uint64_t)(SCALE_BASE + block);
}

static uint64_t scale_write(
    int block, int col, uint32_t carrier
) {
    uint64_t status;
    ROCC_INSTRUCTION_R_R_R(
        3, status, key(block, col),
        (uint64_t)carrier, 23
    );
    return status;
}

static uint64_t scale_release(int block, int col) {
    uint64_t status;
    ROCC_INSTRUCTION_R_R_R(
        3, status, key(block, col), 0ULL, 24
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

static void config_gemmini(void) {
    for (int id = 0; id < 2; ++id) {
        uint64_t rs1 =
            ((uint64_t)DIM << 16) |
            (1ULL << 8) |
            ((uint64_t)id << 3) |
            1ULL;

        // A has 32 elements per row.
        // B has 16 elements per row.
        uint64_t stride = (id == 0) ? KDIM : DIM;

        ROCC_INSTRUCTION_0_R_R(3, rs1, stride, 0);
    }

    // WS, A/C local stride 1.
    ROCC_INSTRUCTION_0_R_R(
        3, (1ULL << 16) | (1ULL << 2),
        1ULL << 48, 0
    );

    // INT32 full-width output.
    ROCC_INSTRUCTION_0_R_R(
        3, 2ULL, DIM * sizeof(int32_t), 0
    );
}

static void issue_gemm(void) {
    uint64_t bounds =
        ((uint64_t)K_TILES << 32) |
        (1ULL << 16) |
        1ULL;

    // I=1, J=1, K=2 physical tiles.
    ROCC_INSTRUCTION_0_R_R(
        3, 0ULL, bounds, 9
    );

    ROCC_INSTRUCTION_0_R_R(
        3,
        (uint64_t)(uintptr_t)&A[0][0],
        (uint64_t)(uintptr_t)&B[0][0],
        10
    );

    ROCC_INSTRUCTION_0_R_R(
        3, 0ULL,
        (uint64_t)(uintptr_t)&C[0][0],
        11
    );

    // A: two K tiles per row.
    // B: one N tile per row.
    ROCC_INSTRUCTION_0_R_R(
        3, K_TILES, DIM, 12
    );

    ROCC_INSTRUCTION_0_R_R(
        3, DIM, DIM, 13
    );

    uint64_t loop_rs1 =
        (1ULL << 1) |
        (1ULL << 20) |
        ((uint64_t)SCALE_BASE << 21) |
        ((uint64_t)SCALE_GEN << 37) |
        ((uint64_t)PASSES_PER_BLOCK << 45);

    ROCC_INSTRUCTION_0_R_R(
        3, loop_rs1, 0ULL, 8
    );
}

static int32_t raw_partial(int pass, int row, int col) {
    int32_t result = 0;

    for (int k = pass * DIM;
         k < (pass + 1) * DIM; ++k) {
        result += (int32_t)A[row][k] *
                  (int32_t)B[k][col];
    }

    return result;
}

static int32_t cpu_reference(int row, int col) {
    int64_t result = 0;

    for (int pass = 0; pass < K_TILES; ++pass) {
        int block = pass / PASSES_PER_BLOCK;
        int32_t partial = raw_partial(pass, row, col);
        uint32_t carrier = carrier_for(block, col);

        if (carrier != ZERO_CARRIER)
            result += (int64_t)partial * (1LL << carrier);
    }

    return (int32_t)result;
}

int main(void) {
    printf(
        "POTAL_D32_K64 starting passesPerBlock=%d\n",
        PASSES_PER_BLOCK
    );

    for (int i = 0; i < DIM; ++i)
        for (int k = 0; k < KDIM; ++k)
            A[i][k] = (int8_t)(
                ((i * 7 + k * 5) % 13) - 6
            );

    for (int k = 0; k < KDIM; ++k)
        for (int j = 0; j < DIM; ++j)
            B[k][j] = (int8_t)(
                ((k * 11 + j * 3) % 17) - 8
            );

    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
            C[i][j] = -9999;

    asm volatile("fence rw,rw" ::: "memory");

    config_gemmini();

    for (int block = 0; block < SCALE_BLOCKS; ++block) {
        for (int col = 0; col < DIM; ++col) {
            uint64_t status =
                scale_write(block, col, carrier_for(block, col));

            if (status != 0) {
                printf(
                    "FAIL scale block=%d col=%d status=%d\n",
                    block, col, (int)status
                );
                exit(1);
            }
        }
    }

    printf("PASS scale initialization\n");

    issue_gemm();

    printf("K32 GEMM issued, waiting for fence\n");

    if (potal_fence() != 0) {
        printf("FAIL fence\n");
        exit(1);
    }

    printf("PASS K32 fence\n");

    asm volatile("fence rw,rw" ::: "memory");

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            int32_t expected = cpu_reference(i, j);
            int32_t actual = C[i][j];

            if (actual != expected) {
                printf(
                    "FAIL C[%d][%d] expected=%d actual=%d "
                    "raw0=%d raw1=%d\n",
                    i, j, expected, actual,
                    raw_partial(0, i, j),
                    raw_partial(1, i, j)
                );
                exit(1);
            }
        }
    }

    printf("PASS K32 numerical comparison\n");

    for (int block = 0; block < SCALE_BLOCKS; ++block) {
        for (int col = 0; col < DIM; ++col) {
            if (scale_release(block, col) != 0) {
                printf(
                    "FAIL release block=%d col=%d\n",
                    block, col
                );
                exit(1);
            }
        }
    }

    printf("POTAL_D32_K64 PASS\n");
    exit(0);
}

