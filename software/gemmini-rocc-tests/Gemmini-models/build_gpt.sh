set -x
cd src

# Build for -linux
riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -c gpt2.c -o gpt2-linux.o

#riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -c gpt2_params.c -o gpt2_params-linux.o

riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -o gpt2-linux gpt2-linux.o gpt2_params-linux.o

rm /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/overlay/root/gpt2-linux
cp gpt2-linux /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/overlay/root


# Build for -pk
riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -static -DBAREMETAL=1 -c gpt2.c -o gpt2-pk.o

# riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -static -DBAREMETAL=1 -c gpt2_params.c -o gpt2_params-pk.o

riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -static -DBAREMETAL=1 -o gpt2-pk gpt2-pk.o gpt2_params-pk.o

#spike --extension=gemmini pk ./src/gpt2-pk ws













# Build Baremetal
#riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -nostdlib -nostartfiles -static -T /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1 gpt2_params.c  -o gpt2_params-baremetal.o \
#        /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/syscalls.c /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/crt.S
#
#riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -nostdlib -nostartfiles -static -T /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1  gpt2.c  -o gpt2-baremetal.o \
#        /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/syscalls.c /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/crt.S
#



#riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/.. -I/home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common -DID_STRING= -Wno-incompatible-pointer-types -nostdlib -nostartfiles -static -T /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1 -o gpt2-baremetal gpt2-baremetal.o gpt2_params-baremetal.o \
#        /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/syscalls.c /home/alveo/firesim/target-design/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common/crt.S
#
