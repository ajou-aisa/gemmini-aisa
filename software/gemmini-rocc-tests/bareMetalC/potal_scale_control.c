
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocc-software/src/xcustom.h"

#define POTAL_SCALE_WRITE   23
#define POTAL_SCALE_RELEASE 24
#define POTAL_FENCE         25

#define STATUS_OK               0
#define STATUS_INVALID_COLUMN   1
#define STATUS_INVALID_SCALE    2
#define STATUS_SCALE_CONFLICT   3
#define STATUS_RELEASE_MISS     4
#define STATUS_INVALID_ENCODING 6

static inline uint64_t scale_key(
    uint8_t slot,
    uint8_t generation,
    uint8_t column
) {
    return ((uint64_t)slot)
        | ((uint64_t)generation << 8)
        | ((uint64_t)column << 16);
}

static inline uint64_t potal_write_raw(
    uint64_t rs1,
    uint64_t carrier
) {
    uint64_t result;
    ROCC_INSTRUCTION_R_R_R(
        3, result, rs1, carrier, 23
    );
    return result;
}

static inline uint64_t potal_write(
    uint8_t slot,
    uint8_t generation,
    uint8_t column,
    uint32_t carrier
) {
    return potal_write_raw(
        scale_key(slot, generation, column),
        carrier
    );
}

static inline uint64_t potal_release(
    uint8_t slot,
    uint8_t generation,
    uint8_t column
) {
    uint64_t result;
    uint64_t rs1 = scale_key(
        slot, generation, column
    );

    ROCC_INSTRUCTION_R_R_R(
        3, result, rs1, 0, 24
    );
    return result;
}

static inline uint64_t potal_fence(void) {
    uint64_t result;

    ROCC_INSTRUCTION_R_R_R(
        3, result, 0, 0, 25
    );
    return result;
}

static void check(
    const char *name,
    uint64_t actual,
    uint64_t expected
) {
    if (actual != expected) {
        printf(
            "FAIL %s: got=%d expected=%d\n",
            name, (int)actual, (int)expected
        );
        exit(1);
    }

    printf("PASS %s\n", name);
}

int main(void) {
    printf("PoTal RoCC control test starting\n");

    // Normal scale allocation.
    check("write shift carrier",
          potal_write(3, 7, 0, 3),
          STATUS_OK);

    // Independent column, same slot.
    check("write zero carrier",
          potal_write(3, 7, 15, 0x80000000U),
          STATUS_OK);

    // Existing scale must not be overwritten.
    check("duplicate write",
          potal_write(3, 8, 0, 4),
          STATUS_SCALE_CONFLICT);

    // Generation-qualified release.
    check("wrong generation release",
          potal_release(3, 8, 0),
          STATUS_RELEASE_MISS);

    check("correct generation release",
          potal_release(3, 7, 0),
          STATUS_OK);

    // Slot reuse after release.
    check("reuse slot",
          potal_write(3, 9, 0, 7),
          STATUS_OK);

    // Invalid carrier must be rejected.
    check("invalid carrier",
          potal_write(4, 1, 0, 0xffffffffU),
          STATUS_INVALID_SCALE);

    // DIM16 has columns 0..15.
    check("invalid column",
          potal_write(6, 1, 16, 1),
          STATUS_INVALID_COLUMN);

    // Reserved rs1 bit is not allowed.
    check("invalid encoding",
          potal_write_raw(
              scale_key(5, 1, 0) | (1ULL << 22),
              1
          ),
          STATUS_INVALID_ENCODING);

    check("fence",
          potal_fence(),
          STATUS_OK);

    // Release remaining allocated entries.
    check("release reused slot",
          potal_release(3, 9, 0),
          STATUS_OK);

    check("release zero carrier",
          potal_release(3, 7, 15),
          STATUS_OK);

    check("final fence",
          potal_fence(),
          STATUS_OK);

    printf("POTAL_SCALE_CONTROL PASS\n");
    exit(0);
}

