from driver import *
from ISA import *

Driver = driver()
datas = Driver.translate("tiled_matmul_ws_full-baremetal.txt", "../../sims/verilator/output/chipyard.TestHarness.CustomGemminiSoCConfig/tiled_matmul_ws-baremetal.out")
#datas = Driver.translate("conv_full-baremetal.txt", "../../sims/verilator/output/chipyard.TestHarness.CustomGemminiSoCConfig/conv-baremetal.out")

datas

Driver.visualize(datas)
plt.show()

Driver.write("tiled_matmul_ws_full-baremetal.txt", datas)
