
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocc-software/src/xcustom.h"

#define DIM 16
#define NTILES 8

static uint8_t input[NTILES][DIM][DIM]
    __attribute__((aligned(64)));

static volatile uint8_t output[NTILES][DIM][DIM]
    __attribute__((aligned(64)));

static void configure_dma(void) {
    // CONFIG_LOAD: block stride DIM, pixel repeat 1
    uint64_t ld_rs1 =
        ((uint64_t)DIM << 16) | (1ULL << 8) | 1ULL;

    ROCC_INSTRUCTION_0_R_R(3, ld_rs1, DIM, 0);

    // CONFIG_STORE: output stride DIM bytes
    ROCC_INSTRUCTION_0_R_R(3, 2ULL, DIM, 0);
}

static void issue_dma(int n) {
    uint64_t desc =
        ((uint64_t)DIM << 48) |
        ((uint64_t)DIM << 32) |
        (uint64_t)(n * DIM);

    ROCC_INSTRUCTION_0_R_R(
        3, (uint64_t)(uintptr_t)&input[n][0][0],
        desc, 2
    );

    ROCC_INSTRUCTION_0_R_R(
        3, (uint64_t)(uintptr_t)&output[n][0][0],
        desc, 3
    );
}

static uint64_t potal_write(void) {
    uint64_t result;
    uint64_t key = (7ULL << 8) | 3ULL;

    ROCC_INSTRUCTION_R_R_R(
        3, result, key, 1ULL, 23
    );
    return result;
}

static uint64_t potal_release(void) {
    uint64_t result;
    uint64_t key = (7ULL << 8) | 3ULL;

    ROCC_INSTRUCTION_R_R_R(
        3, result, key, 0ULL, 24
    );
    return result;
}

static uint64_t potal_fence(void) {
    uint64_t result;

    ROCC_INSTRUCTION_R_R_R(
        3, result, 0ULL, 0ULL, 25
    );
    return result;
}

static void verify(int begin, int end) {
    asm volatile("fence rw,rw" ::: "memory");

    for (int n = begin; n < end; ++n) {
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                if (input[n][i][j] != output[n][i][j]) {
                    printf(
                        "FAIL tile=%d row=%d col=%d\n",
                        n, i, j
                    );
                    exit(1);
                }
            }
        }
    }
}

int main(void) {
    printf("PoTal DMA fence test starting\n");

    for (int n = 0; n < NTILES; ++n)
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                input[n][i][j] =
                    (uint8_t)((n * 37 + i * 11 + j * 5 + 1) & 255);

    asm volatile("fence rw,rw" ::: "memory");

    configure_dma();

    if (potal_write() != 0) {
        printf("FAIL scale write\n");
        exit(1);
    }

    // Batch 1: verify POTAL_FENCE waits for DMA.
    for (int n = 0; n < 4; ++n)
        issue_dma(n);

    if (potal_fence() != 0) {
        printf("FAIL DMA fence\n");
        exit(1);
    }

    verify(0, 4);
    printf("PASS DMA after POTAL_FENCE\n");

    // Batch 2: verify RELEASE waits for DMA.
    for (int n = 4; n < NTILES; ++n)
        issue_dma(n);

    if (potal_release() != 0) {
        printf("FAIL scale release\n");
        exit(1);
    }

    verify(4, NTILES);
    printf("PASS DMA after POTAL_RELEASE\n");

    printf("POTAL_FENCE_DMA PASS\n");
    exit(0);
}

