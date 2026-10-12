#include <stdio.h>
#include <stdint.h>

#define read_csr(reg) ({ \
    uint64_t __tmp; \
    asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
    __tmp; \
})

#define write_csr(reg, val) ({ \
    asm volatile ("csrw " #reg ", %0" :: "r"(val)); \
})

int main() {
    // Set event ID for mhpmcounter3 to track branch mispredictions (e.g., 0x3)
    write_csr(mhpmevent3, 0x3);

    uint64_t before = read_csr(mhpmcounter3);

    // Code that involves branching
    volatile int x = 0;
    for (int i = 0; i < 100000000; i++) {
        if ((i % 3) == 0) {
            x++;
        } else {
            x--;
        }
    }

    uint64_t after = read_csr(mhpmcounter3);
    printf("Branch mispredictions: %lu\n", after - before);

    return 0;
}

