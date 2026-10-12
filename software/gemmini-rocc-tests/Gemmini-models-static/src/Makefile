# Compiler and Flags for pk and linux
CC_PK = riscv64-unknown-linux-gnu-gcc
CC_LINUX = riscv64-unknown-linux-gnu-gcc

# Compiler and Flags for baremetal
CC_BAREMETAL = riscv64-unknown-elf-gcc

# Common Compilation Flags
COMMON_FLAGS = -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -DID_STRING= -DPRINT_TILE=0

# Compilation Flags for pk
CFLAGS_PK = $(COMMON_FLAGS) -static -DBAREMETAL=1

# Compilation Flags for baremetal
CFLAGS_BAREMETAL = $(COMMON_FLAGS) -DBAREMETAL=1 -nostdlib -nostartfiles

# Linking Flags
LDFLAGS_PK = -lm -lgcc
LDFLAGS_BAREMETAL = -lm -lgcc -static -T ../include/riscv/test.ld
LDFLAGS_LINUX = -lm -lgcc

# Source and Target Executables
SRCS = gpt2.c
TARGET_PK = gpt2-pk
TARGET_BAREMETAL = gpt2-baremetal
TARGET_LINUX = gpt2-linux

# Default Target
all: pk baremetal linux

# PK Target
pk:
	$(CC_PK) $(CFLAGS_PK) $(SRCS) -o $(TARGET_PK) $(LDFLAGS_PK)

# Baremetal Target
baremetal:
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $(SRCS) ../include/riscv/syscalls.c ../include/riscv/crt.S -o $(TARGET_BAREMETAL) $(LDFLAGS_BAREMETAL)

# Linux Target
linux:
	$(CC_LINUX) $(COMMON_FLAGS) $(SRCS) -o $(TARGET_LINUX) $(LDFLAGS_LINUX)

# Clean up build artifacts
clean:
	rm -f $(TARGET_PK) $(TARGET_BAREMETAL) $(TARGET_LINUX)

# Specify phony targets
.PHONY: all pk baremetal linux clean
