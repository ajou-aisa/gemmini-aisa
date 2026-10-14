
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocc-software/src/xcustom.h"

// ============================================================
// PoTal Sat32 end-to-end regression
//
// DIM16 / A8W8 / WS / Sat32 ACC
// M=16, N=16, K=48
// I=1, J=1, K_TILES=3
//
// Tests:
// - Positive/negative SCU saturation
// - Positive/negative ACC saturation
// - Saturation followed by opposite-sign accumulation
// - Exact INT32 boundaries
// - ZeroCarrier and large shift
// ============================================================

#define DIM 64
#define K_TILES 3
#define KDIM (DIM * K_TILES)

#define SCALE_BASE 8
#define SCALE_GEN 7
#define ZERO_CARRIER 0x80000000U

#define PASSES_PER_BLOCK 1
#if DIM == 16
#define SHIFT_ADJUST 0
#elif DIM == 32
#define SHIFT_ADJUST 1
#elif DIM == 64
#define SHIFT_ADJUST 2
#else
#error Unsupported DIM
#endif

static int8_t A[DIM][KDIM]
    __attribute__((aligned(64)));

static int8_t B[KDIM][DIM]
    __attribute__((aligned(64)));

static volatile int32_t C[DIM][DIM]
    __attribute__((aligned(64)));

typedef struct {
    unsigned scu_pos;
    unsigned scu_neg;
    unsigned acc_pos;
    unsigned acc_neg;
    unsigned recede;
} SatStats;

// ============================================================
// Scale selection
// ============================================================

static uint32_t carrier_for(int pass, int col) {
    static const uint32_t carriers[16] = {
        28 - SHIFT_ADJUST, 28 - SHIFT_ADJUST,
        25 - SHIFT_ADJUST, 25 - SHIFT_ADJUST,
        28 - SHIFT_ADJUST, 26 - SHIFT_ADJUST,
        ZERO_CARRIER, 0,
        31, 32, 15, 24 - SHIFT_ADJUST,
        30, 30, 30, 31
    };
    int lane = col % 16;
    if (lane == 15 && pass == 2)
        return 30;
    return carriers[lane];
}

// ============================================================
// Test input generation
// ============================================================

static int8_t b_pattern(
    int pass, int local_k, int col
) {
    switch (col % 16) {
        case 0: return pass == 0 ? 1 : 0;
        case 1: return pass == 0 ? -1 : 0;

        // ACC overflow, then reverse direction.
        case 2: return pass < 2 ? 3 : -3;
        case 3: return pass < 2 ? -3 : 3;

        // SCU saturation and cancellation.
        case 4:
        case 5:
            return pass == 0 ? 1 :
                   pass == 1 ? -1 : 0;

        case 6: return 3;

        case 7:
            return pass == 0 ? 2 :
                   pass == 1 ? 1 : -1;

        case 8: return 0;
        case 9: return pass == 0 ? 1 : 0;

        case 10:
            return pass == 0 ? 3 :
                   pass == 1 ? -2 : 1;

        case 11: return 4;

        // Produce raw partial sums of exactly +/-1.
        case 12:
            return local_k == 0 ? 1 : 0;

        case 13:
            return local_k == 0 ? -1 : 0;

        case 14:
            if (local_k != 0) return 0;
            return pass == 0 ? 1 :
                   pass == 1 ? -1 : 0;

        case 15:
            if (local_k != 0) return 0;
            return pass == 0 ? -1 :
                   pass == 2 ? 1 : 0;
    }

    return 0;
}

// ============================================================
// Scale commands
// ============================================================

static uint64_t scale_key(int pass, int col) {
    return ((uint64_t)col << 16) |
           ((uint64_t)SCALE_GEN << 8) |
           (uint64_t)(SCALE_BASE + pass);
}

static uint64_t scale_write(
    int pass, int col, uint32_t carrier
) {
    uint64_t status;

    ROCC_INSTRUCTION_R_R_R(
        3, status, scale_key(pass, col),
        (uint64_t)carrier, 23
    );

    return status;
}

static uint64_t scale_release(int pass, int col) {
    uint64_t status;

    ROCC_INSTRUCTION_R_R_R(
        3, status, scale_key(pass, col),
        0ULL, 24
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

// ============================================================
// Gemmini configuration
// ============================================================

static void config_gemmini(void) {
    for (int id = 0; id < 2; ++id) {
        uint64_t rs1 =
            ((uint64_t)DIM << 16) |
            (1ULL << 8) |
            ((uint64_t)id << 3) |
            1ULL;

        uint64_t stride =
            id == 0 ? KDIM : DIM;

        ROCC_INSTRUCTION_0_R_R(
            3, rs1, stride, 0
        );
    }

    // WS, local strides 1.
    ROCC_INSTRUCTION_0_R_R(
        3,
        (1ULL << 16) | (1ULL << 2),
        1ULL << 48,
        0
    );

    // Full-width INT32 output.
    ROCC_INSTRUCTION_0_R_R(
        3, 2ULL,
        DIM * sizeof(int32_t), 0
    );
}

static void issue_gemm(void) {
    uint64_t bounds =
        ((uint64_t)K_TILES << 32) |
        (1ULL << 16) |
        1ULL;

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

    // Row strides in elements.
    // A=48, B=16, D=16, C=16.
    ROCC_INSTRUCTION_0_R_R(
        3, KDIM, DIM, 12
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

// ============================================================
// Independent saturating CPU reference
// ============================================================

static int32_t sat32(
    int64_t value,
    unsigned *positive,
    unsigned *negative
) {
    if (value > INT32_MAX) {
        (*positive)++;
        return INT32_MAX;
    }

    if (value < INT32_MIN) {
        (*negative)++;
        return INT32_MIN;
    }

    return (int32_t)value;
}

static int32_t raw_partial(
    int pass, int row, int col
) {
    int32_t result = 0;

    for (int k = pass * DIM;
         k < (pass + 1) * DIM; ++k) {

        result +=
            (int32_t)A[row][k] *
            (int32_t)B[k][col];
    }

    return result;
}

static int32_t scu_reference(
    int32_t raw,
    uint32_t carrier,
    SatStats *stats
) {
    if (carrier == ZERO_CARRIER || raw == 0)
        return 0;

    if (carrier >= 32) {
        if (raw > 0) {
            stats->scu_pos++;
            return INT32_MAX;
        }

        stats->scu_neg++;
        return INT32_MIN;
    }

    int64_t scaled =
        (int64_t)raw * (1LL << carrier);

    return sat32(
        scaled,
        &stats->scu_pos,
        &stats->scu_neg
    );
}

static int32_t cpu_reference(
    int row, int col, SatStats *stats
) {
    int32_t acc = 0;
    int early_acc_saturation = 0;

    for (int pass = 0; pass < K_TILES; ++pass) {
        int32_t raw =
            raw_partial(pass, row, col);

        int32_t partial =
            scu_reference(
                raw,
                carrier_for(pass, col),
                stats
            );

        int64_t wide_sum =
            (int64_t)acc + (int64_t)partial;

        int overflow =
            wide_sum > INT32_MAX ||
            wide_sum < INT32_MIN;

        int32_t next = sat32(
            wide_sum,
            &stats->acc_pos,
            &stats->acc_neg
        );

        if (pass == K_TILES - 1 &&
            early_acc_saturation &&
            next != acc) {
            stats->recede++;
        }

        if (overflow && pass < K_TILES - 1)
            early_acc_saturation = 1;

        acc = next;
    }

    return acc;
}

// ============================================================
// Main test
// ============================================================

int main(void) {
    printf("POTAL_SAT32_D64 starting K=%d\n", KDIM);

    // Alternate signs across output rows.
    for (int i = 0; i < DIM; ++i) {
        for (int k = 0; k < KDIM; ++k) {
            A[i][k] = (i & 1) ? -1 : 1;
        }
    }

    for (int k = 0; k < KDIM; ++k) {
        int pass = k / DIM;
        int local_k = k % DIM;

        for (int j = 0; j < DIM; ++j) {
            B[k][j] =
                b_pattern(pass, local_k, j);
        }
    }

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            C[i][j] = -9999;
        }
    }

    // Sanity-check the reference vectors.
    SatStats sanity = {0};

    if (cpu_reference(0, 2, &sanity) != 536870911 ||
        cpu_reference(0, 3, &sanity) != -536870912 ||
        cpu_reference(0, 4, &sanity) != -1) {
        printf("FAIL reference vector construction\n");
        exit(1);
    }

    asm volatile("fence rw,rw" ::: "memory");

    config_gemmini();

    // One independent scale slot per K pass.
    for (int pass = 0; pass < K_TILES; ++pass) {
        for (int col = 0; col < DIM; ++col) {
            uint64_t status = scale_write(
                pass, col,
                carrier_for(pass, col)
            );

            if (status != 0) {
                printf(
                    "FAIL scale write pass=%d "
                    "col=%d status=%d\n",
                    pass, col, (int)status
                );
                exit(1);
            }
        }
    }

    printf("PASS scale initialization\n");

    issue_gemm();

    printf("Sat32 GEMM issued; waiting for fence\n");

    if (potal_fence() != 0) {
        printf("FAIL POTAL_FENCE\n");
        exit(1);
    }

    printf("PASS POTAL_FENCE\n");

    asm volatile("fence rw,rw" ::: "memory");

    SatStats stats = {0};
    int checked = 0;

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            int32_t expected =
                cpu_reference(i, j, &stats);

            int32_t actual = C[i][j];

            if (actual != expected) {
                printf(
                    "FAIL C[%d][%d] expected=%d "
                    "actual=%d raw0=%d "
                    "raw1=%d raw2=%d\n",
                    i, j, expected, actual,
                    raw_partial(0, i, j),
                    raw_partial(1, i, j),
                    raw_partial(2, i, j)
                );
                exit(1);
            }

            checked++;
        }
    }

    printf(
        "PASS Sat32 numerical comparison "
        "(%d outputs)\n",
        checked
    );

    printf(
        "Saturation coverage: "
        "SCU(+%u,-%u) ACC(+%u,-%u) "
        "recede=%u\n",
        stats.scu_pos,
        stats.scu_neg,
        stats.acc_pos,
        stats.acc_neg,
        stats.recede
    );

    if (stats.scu_pos == 0 ||
        stats.scu_neg == 0 ||
        stats.acc_pos == 0 ||
        stats.acc_neg == 0 ||
        stats.recede == 0) {
        printf("FAIL saturation coverage\n");
        exit(1);
    }

    printf("PASS saturation coverage\n");

    for (int pass = 0; pass < K_TILES; ++pass) {
        for (int col = 0; col < DIM; ++col) {
            if (scale_release(pass, col) != 0) {
                printf(
                    "FAIL scale release "
                    "pass=%d col=%d\n",
                    pass, col
                );
                exit(1);
            }
        }
    }

    printf("PASS scale release\n");
    printf("POTAL_SAT32_D64 PASS\n");

    exit(0);
}

