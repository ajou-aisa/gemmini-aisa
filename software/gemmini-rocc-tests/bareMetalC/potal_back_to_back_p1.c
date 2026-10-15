
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
#define PASSES_PER_BLOCK 1

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

static int8_t A[2][MDIM][KDIM]
    __attribute__((aligned(64)));

static int8_t B[2][KDIM][NDIM]
    __attribute__((aligned(64)));

static volatile int32_t C[2][MDIM][NDIM]
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

static void issue_gemm(int job) {

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
        (uint64_t)(uintptr_t)&A[job][0][0],
        (uint64_t)(uintptr_t)&B[job][0][0],
        10
    );

    // LOOP_WS_CONFIG_ADDRS_DC
    //
    // No bias (D=0)
    // Full-width INT32 output C

    ROCC_INSTRUCTION_0_R_R(
        3,
        0ULL,
        (uint64_t)(uintptr_t)&C[job][0][0],
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
    int job,
    int pass,
    int row,
    int col
) {
    int32_t result = 0;

    int k_begin = pass * DIM;
    int k_end = (pass + 1) * DIM;

    for (int k = k_begin; k < k_end; ++k) {
        result +=
            (int32_t)A[job][row][k] *
            (int32_t)B[job][k][col];
    }

    return result;
}

static int32_t cpu_reference(
    int job,
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
            raw_partial(job, pass, row, col);

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
    const int jobs = 2;
    int32_t expected_first[2], expected_last[2];
    uint64_t checksums[2] = {0, 0};

    printf("POTAL_BACK_TO_BACK starting passesPerBlock=%d\n",
           PASSES_PER_BLOCK);

    // Distinct A/B/C buffers for each GEMM.
    for (int job = 0; job < jobs; ++job) {
        for (int i = 0; i < MDIM; ++i)
            for (int k = 0; k < KDIM; ++k)
                A[job][i][k] = (int8_t)(
                    ((i * (7 + job) + k * (5 + 2 * job) +
                      3 * job) % 13) - 6);

        for (int k = 0; k < KDIM; ++k)
            for (int j = 0; j < NDIM; ++j)
                B[job][k][j] = (int8_t)(
                    ((k * (11 + job) + j * (3 + job) +
                      5 * job) % 17) - 8);

        for (int i = 0; i < MDIM; ++i)
            for (int j = 0; j < NDIM; ++j)
                C[job][i][j] =
                    -15000 - 1000 * job - i * NDIM - j;

        // Compute boundary references before issuing either GEMM.
        expected_first[job] = cpu_reference(job, 0, 0);
        expected_last[job] =
            cpu_reference(job, MDIM - 1, NDIM - 1);
    }

    asm volatile("fence rw,rw" ::: "memory");
    config_gemmini();

    // Both GEMMs use the same Scale Generation and slots.
    for (int block = 0; block < SCALE_BLOCKS; ++block)
        for (int tile = 0; tile < N_TILES; ++tile)
            for (int col = 0; col < DIM; ++col) {
                uint64_t status = scale_write(
                    block, tile, col,
                    carrier_for(block, tile, col));

                if (status != 0) {
                    printf(
                        "FAIL Scale WRITE "
                        "block=%d tile=%d col=%d status=%llu\n",
                        block, tile, col,
                        (unsigned long long)status);
                    exit(1);
                }
            }

    // =====================================================
    // Back-to-back command submission.
    //
    // No POTAL_FENCE, RELEASE, printf, or CPU reference work
    // between the two GEMM issues.
    // =====================================================

    uint64_t t0 = read_cycles();

    issue_gemm(0);

    uint64_t t1 = read_cycles();

    issue_gemm(1);

    uint64_t t2 = read_cycles();

    // One accelerator fence waits for both jobs.
    uint64_t status = potal_fence();
    uint64_t t3 = read_cycles();

    asm volatile("fence rw,rw" ::: "memory");

    // Immediate boundary verification, before printf.
    int32_t first0 = C[0][0][0];
    int32_t last0 = C[0][MDIM - 1][NDIM - 1];
    int32_t first1 = C[1][0][0];
    int32_t last1 = C[1][MDIM - 1][NDIM - 1];

    if (status != 0 ||
        first0 != expected_first[0] ||
        last0 != expected_last[0] ||
        first1 != expected_first[1] ||
        last1 != expected_last[1]) {

        printf(
            "FAIL FENCE/immediate output status=%llu\n",
            (unsigned long long)status);

        printf(
            "GEMM0 first=%d/%d last=%d/%d\n",
            first0, expected_first[0],
            last0, expected_last[0]);

        printf(
            "GEMM1 first=%d/%d last=%d/%d\n",
            first1, expected_first[1],
            last1, expected_last[1]);

        exit(1);
    }

    printf("PASS immediate FENCE verification\n");

    // Full numerical comparison: 1024 outputs per GEMM.
    for (int job = 0; job < jobs; ++job) {
        int checked = 0;

        for (int i = 0; i < MDIM; ++i)
            for (int j = 0; j < NDIM; ++j) {
                int32_t expected = cpu_reference(job, i, j);
                int32_t actual = C[job][i][j];

                if (actual != expected) {
                    printf(
                        "FAIL job=%d C[%d][%d] "
                        "expected=%d actual=%d\n",
                        job, i, j, expected, actual);
                    exit(1);
                }

                checksums[job] =
                    checksums[job] * 131ULL + (uint32_t)actual;

                ++checked;
            }

        if (checked != MDIM * NDIM)
            exit(1);

        printf(
            "PASS job=%d outputs=%d checksum=%llu\n",
            job, checked,
            (unsigned long long)checksums[job]);
    }

    // Detect accidentally identical outputs.
    if (checksums[0] == checksums[1]) {
        printf("FAIL two outputs have identical checksums\n");
        exit(1);
    }

    // Scale ownership ends only after both jobs drain.
    for (int block = 0; block < SCALE_BLOCKS; ++block)
        for (int tile = 0; tile < N_TILES; ++tile)
            for (int col = 0; col < DIM; ++col) {
                status = scale_release(block, tile, col);

                if (status != 0) {
                    printf(
                        "FAIL Scale RELEASE "
                        "block=%d tile=%d col=%d status=%llu\n",
                        block, tile, col,
                        (unsigned long long)status);
                    exit(1);
                }
            }

    printf(
        "CYCLES issue0=%llu issue1=%llu final_fence=%llu\n",
        (unsigned long long)(t1 - t0),
        (unsigned long long)(t2 - t1),
        (unsigned long long)(t3 - t2));

    printf(
        "POTAL_BACK_TO_BACK_P%d PASS\n",
        PASSES_PER_BLOCK);

    exit(0);
}
