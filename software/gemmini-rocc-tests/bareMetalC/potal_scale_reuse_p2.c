
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocc-software/src/xcustom.h"


static inline uint64_t read_cycles(void) {
    uint64_t value;
    asm volatile("rdcycle %0" : "=r"(value));
    return value;
}

// ============================================================
// PoTal multi-output-tile GEMM test
//
// Hardware: DIM16, A8W8, WS, Sat32 ACC
// GEMM:     M=16, N=32, K=32
//
// I=1, J=2, K=2 physical tiles
//
// Case A: PASSES_PER_BLOCK=2
//   Block 0, J0 -> Slot 8
//   Block 0, J1 -> Slot 9
//
// Case B: PASSES_PER_BLOCK=1
//   Block 0, J0 -> Slot 8
//   Block 0, J1 -> Slot 9
//   Block 1, J0 -> Slot 10
//   Block 1, J1 -> Slot 11
// ============================================================

#define DIM 16

#define M_TILES 2
#define MDIM (DIM * M_TILES)

#define N_TILES 2
#define NDIM (DIM * N_TILES)

#define K_TILES 2
#define KDIM (DIM * K_TILES)

#define SCALE_BASE 8
static uint32_t scale_generation = 1;

#define ZERO_CARRIER 0x80000000U

// Case A: 2 (shared K32 scale)
// Case B: 1 (independent K16 scales)
#define PASSES_PER_BLOCK 2

#if (PASSES_PER_BLOCK != 1) && (PASSES_PER_BLOCK != 2)
#error Unsupported PASSES_PER_BLOCK
#endif

#if K_TILES % PASSES_PER_BLOCK != 0
#error Invalid scale block configuration
#endif

#define SCALE_BLOCKS (K_TILES / PASSES_PER_BLOCK)

// ============================================================
// Matrices
// ============================================================

static int8_t A[MDIM][KDIM]
    __attribute__((aligned(64)));

static int8_t B[KDIM][NDIM]
    __attribute__((aligned(64)));

static volatile int32_t C[MDIM][NDIM]
    __attribute__((aligned(64)));

// ============================================================
// Scale carrier and slot mapping
//
// Slot = SCALE_BASE + block * N_TILES + tile
//
// Hardware column is local_col (0..15), not global N column.
// ============================================================

static uint32_t carrier_for(
    int block,
    int tile,
    int local_col
) {
    // Give each N tile a different scale pattern.
    // This detects incorrect scale-slot selection.
    switch ((local_col + 2 * block + 3 * tile + scale_generation) % 4) {
        case 0:
            return 0;             // x1

        case 1:
            return 1;             // x2

        case 2:
            return ZERO_CARRIER;  // x0

        default:
            return 2;             // x4
    }
}

static uint64_t scale_key(
    int block,
    int tile,
    int local_col
) {
    uint64_t slot =
        (uint64_t)SCALE_BASE +
        (uint64_t)block * N_TILES +
        (uint64_t)tile;

    return
        ((uint64_t)local_col << 16) |
        ((uint64_t)scale_generation << 8) |
        slot;
}

// ============================================================
// PoTal RoCC scale commands
// ============================================================

static uint64_t scale_write(
    int block,
    int tile,
    int local_col,
    uint32_t carrier
) {
    uint64_t status;

    ROCC_INSTRUCTION_R_R_R(
        3,
        status,
        scale_key(block, tile, local_col),
        (uint64_t)carrier,
        23
    );

    return status;
}

static uint64_t scale_release(
    int block,
    int tile,
    int local_col
) {
    uint64_t status;

    ROCC_INSTRUCTION_R_R_R(
        3,
        status,
        scale_key(block, tile, local_col),
        0ULL,
        24
    );

    return status;
}

static uint64_t potal_fence(void) {
    uint64_t status;

    ROCC_INSTRUCTION_R_R_R(
        3,
        status,
        0ULL,
        0ULL,
        25
    );

    return status;
}

// ============================================================
// Gemmini configuration
// ============================================================

static void config_gemmini(void) {

    // CONFIG_LOAD
    //
    // A: 16 x 32 INT8, row stride = 32 bytes
    // B: 32 x 32 INT8, row stride = 32 bytes
    //
    // State 0 -> A
    // State 1 -> B

    for (int id = 0; id < 2; ++id) {

        uint64_t rs1 =
            ((uint64_t)DIM << 16) |
            (1ULL << 8) |
            ((uint64_t)id << 3) |
            1ULL;

        uint64_t stride =
            (id == 0) ? KDIM : NDIM;

        ROCC_INSTRUCTION_0_R_R(
            3,
            rs1,
            stride,
            0
        );
    }

    // CONFIG_EX
    //
    // WS dataflow
    // No activation
    // Input shift = 0
    // Local A stride = 1
    // Local C stride = 1

    uint64_t ex_rs1 =
        (1ULL << 16) |
        (1ULL << 2);

    uint64_t ex_rs2 =
        (1ULL << 48);

    ROCC_INSTRUCTION_0_R_R(
        3,
        ex_rs1,
        ex_rs2,
        0
    );

    // CONFIG_STORE
    //
    // C: 16 x 32 INT32
    // DRAM row stride = 32 * 4 = 128 bytes

    ROCC_INSTRUCTION_0_R_R(
        3,
        2ULL,
        NDIM * sizeof(int32_t),
        0
    );
}

// ============================================================
// LOOP_WS GEMM
// ============================================================

static void issue_gemm(void) {

    // LOOP_WS_CONFIG_BOUNDS
    //
    // I=1, J=2, K=2
    // No padding

    uint64_t bounds =
        ((uint64_t)K_TILES << 32) |
        ((uint64_t)N_TILES << 16) |
        (uint64_t)M_TILES;

    ROCC_INSTRUCTION_0_R_R(
        3,
        0ULL,
        bounds,
        9
    );

    // LOOP_WS_CONFIG_ADDRS_AB
    //
    // Activation and weight DRAM addresses

    ROCC_INSTRUCTION_0_R_R(
        3,
        (uint64_t)(uintptr_t)&A[0][0],
        (uint64_t)(uintptr_t)&B[0][0],
        10
    );

    // LOOP_WS_CONFIG_ADDRS_DC
    //
    // No bias (D=0)
    // Full-width INT32 output C

    ROCC_INSTRUCTION_0_R_R(
        3,
        0ULL,
        (uint64_t)(uintptr_t)&C[0][0],
        11
    );

    // LOOP_WS_CONFIG_STRIDES_AB
    //
    // IMPORTANT:
    // LOOP_WS strides are in elements, not tile counts.
    //
    // A row stride = 32 elements
    // B row stride = 32 elements

    ROCC_INSTRUCTION_0_R_R(
        3,
        KDIM,
        NDIM,
        12
    );

    // LOOP_WS_CONFIG_STRIDES_DC
    //
    // D row stride = 32 elements (unused)
    // C row stride = 32 elements

    ROCC_INSTRUCTION_0_R_R(
        3,
        NDIM,
        NDIM,
        13
    );

    // LOOP_WS
    //
    // rs1[1]     = full_C
    // rs1[20]    = PoTal enable
    // rs1[36:21] = scaleBase
    // rs1[44:37] = scaleGeneration
    // rs1[60:45] = passesPerBlock
    //
    // No transpose or resadd.

    uint64_t loop_rs1 =
        (1ULL << 1) |
        (1ULL << 20) |
        ((uint64_t)SCALE_BASE << 21) |
        ((uint64_t)scale_generation << 37) |
        ((uint64_t)PASSES_PER_BLOCK << 45);

    ROCC_INSTRUCTION_0_R_R(
        3,
        loop_rs1,
        0ULL,
        8
    );
}

// ============================================================
// CPU reference
//
// Each physical K pass:
//   1. Calculate INT32 raw partial
//   2. Select the correct scale block and N tile
//   3. Apply PoT shift or zero scale
//   4. Accumulate scaled partials
//
// Input values are small enough to avoid INT32 saturation.
// ============================================================

static int32_t raw_partial(
    int pass,
    int row,
    int col
) {
    int32_t result = 0;

    int k_begin = pass * DIM;
    int k_end = (pass + 1) * DIM;

    for (int k = k_begin; k < k_end; ++k) {
        result +=
            (int32_t)A[row][k] *
            (int32_t)B[k][col];
    }

    return result;
}

static int32_t cpu_reference(
    int row,
    int col
) {
    int64_t result = 0;

    // Convert global N column to output tile
    // and physical array column.
    int tile = col / DIM;
    int local_col = col % DIM;

    for (int pass = 0; pass < K_TILES; ++pass) {

        int block =
            pass / PASSES_PER_BLOCK;

        int32_t partial =
            raw_partial(pass, row, col);

        uint32_t carrier =
            carrier_for(block, tile, local_col);

        if (carrier == ZERO_CARRIER) {
            continue;
        }

        result +=
            (int64_t)partial * (1LL << carrier);
    }

    return (int32_t)result;
}

// ============================================================
// Test
// ============================================================


int main(void) {
    enum {
        ROUNDS = 4,
        STATUS_CONFLICT = 3,
        STATUS_MISS = 4
    };

    uint64_t previous_signature = 0;

    printf(
        "POTAL_SCALE_REUSE starting passesPerBlock=%d rounds=%d\n",
        PASSES_PER_BLOCK, ROUNDS
    );

    config_gemmini();

    for (int round = 0; round < ROUNDS; ++round) {
        uint32_t gen = (uint32_t)(round + 1);
        scale_generation = gen;

        // Change inputs between generations.
        for (int i = 0; i < MDIM; ++i)
            for (int k = 0; k < KDIM; ++k)
                A[i][k] = (int8_t)(
                    ((i * (7 + gen) +
                      k * (5 + 2 * gen) +
                      3 * gen) % 13) - 6
                );

        for (int k = 0; k < KDIM; ++k)
            for (int j = 0; j < NDIM; ++j)
                B[k][j] = (int8_t)(
                    ((k * (11 + gen) +
                      j * (3 + gen) +
                      5 * gen) % 17) - 8
                );

        // Force all outputs to be overwritten.
        for (int i = 0; i < MDIM; ++i)
            for (int j = 0; j < NDIM; ++j)
                C[i][j] =
                    -10001 - 101 * round - i * NDIM - j;

        asm volatile("fence rw,rw" ::: "memory");

        // Allocate the SAME ScaleMemory slots.
        for (int block = 0;
             block < SCALE_BLOCKS;
             ++block) {

            for (int tile = 0;
                 tile < N_TILES;
                 ++tile) {

                for (int col = 0;
                     col < DIM;
                     ++col) {

                    uint64_t status = scale_write(
                        block,
                        tile,
                        col,
                        carrier_for(block, tile, col)
                    );

                    if (status != 0) {
                        printf(
                            "FAIL scale WRITE "
                            "round=%d gen=%u "
                            "block=%d tile=%d col=%d "
                            "status=%llu\n",
                            round, gen,
                            block, tile, col,
                            (unsigned long long)status
                        );
                        exit(1);
                    }
                }
            }
        }

        // An occupied slot must reject another generation.
        scale_generation = gen + 1;

        uint64_t status = scale_write(
            0, 0, 0, 1
        );

        scale_generation = gen;

        if (status != STATUS_CONFLICT) {
            printf(
                "FAIL expected SCALE_CONFLICT "
                "round=%d got=%llu\n",
                round,
                (unsigned long long)status
            );
            exit(1);
        }

        // Execute GEMM and wait for its completion.
        uint64_t t_issue = read_cycles();

        issue_gemm();

        uint64_t issue_cycles =
            read_cycles() - t_issue;

        uint64_t t_fence = read_cycles();

        status = potal_fence();

        uint64_t fence_cycles =
            read_cycles() - t_fence;

        if (status != 0) {
            printf(
                "FAIL FENCE round=%d status=%llu\n",
                round,
                (unsigned long long)status
            );
            exit(1);
        }

        asm volatile("fence rw,rw" ::: "memory");

        // Compare every output against this round's reference.
        uint64_t signature = 0;
        int checked = 0;

        for (int i = 0; i < MDIM; ++i) {
            for (int j = 0; j < NDIM; ++j) {

                int32_t expected = cpu_reference(i, j);
                int32_t actual = C[i][j];

                if (actual != expected) {
                    printf(
                        "FAIL round=%d gen=%u "
                        "C[%d][%d] expected=%d actual=%d\n",
                        round, gen, i, j,
                        expected, actual
                    );
                    exit(1);
                }

                signature =
                    signature * 131u + (uint32_t)actual;

                checked++;
            }
        }

        if (checked != MDIM * NDIM) {
            printf(
                "FAIL incomplete output coverage "
                "round=%d\n",
                round
            );
            exit(1);
        }

        if (round && signature == previous_signature) {
            printf(
                "FAIL unchanged output signature "
                "round=%d\n",
                round
            );
            exit(1);
        }

        previous_signature = signature;

        // Incorrect-generation RELEASE must not free the slot.
        scale_generation = gen + 1;

        status = scale_release(0, 0, 0);

        scale_generation = gen;

        if (status != STATUS_MISS) {
            printf(
                "FAIL expected RELEASE_MISS "
                "round=%d got=%llu\n",
                round,
                (unsigned long long)status
            );
            exit(1);
        }

        // Release all currently allocated entries.
        for (int block = 0;
             block < SCALE_BLOCKS;
             ++block) {

            for (int tile = 0;
                 tile < N_TILES;
                 ++tile) {

                for (int col = 0;
                     col < DIM;
                     ++col) {

                    status = scale_release(
                        block, tile, col
                    );

                    if (status != 0) {
                        printf(
                            "FAIL RELEASE "
                            "round=%d gen=%u "
                            "block=%d tile=%d col=%d "
                            "status=%llu\n",
                            round, gen,
                            block, tile, col,
                            (unsigned long long)status
                        );
                        exit(1);
                    }
                }
            }
        }

        printf(
            "PASS round=%d generation=%u outputs=%d "
            "issue=%llu fence=%llu checksum=%llu\n",
            round, gen, checked,
            (unsigned long long)issue_cycles,
            (unsigned long long)fence_cycles,
            (unsigned long long)signature
        );
    }

    printf(
        "POTAL_SCALE_REUSE_P%d PASS\n",
        PASSES_PER_BLOCK
    );

    exit(0);
}
