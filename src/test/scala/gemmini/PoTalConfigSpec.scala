
package gemmini

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PoTalConfigSpec extends AnyFlatSpec with Matchers {

  behavior of "PoTal A8W8-D16 configuration"

  private val cfg = PoTalConfigs.a8w8D16Sat32

  it should "use the required WS array geometry" in {
    cfg.dataflow shouldBe Dataflow.WS

    cfg.DIM shouldBe 16

    cfg.tileRows shouldBe 1
    cfg.tileColumns shouldBe 1

    cfg.meshRows shouldBe 16
    cfg.meshColumns shouldBe 16

    cfg.inputType.getWidth shouldBe 8
    cfg.spatialArrayOutputType.getWidth shouldBe 20
    cfg.accType.getWidth shouldBe 32
  }

  it should "preserve the intended SP and ACC capacities" in {
    cfg.sp_banks shouldBe 4
    cfg.sp_singleported shouldBe false
    cfg.sp_bank_entries shouldBe 4096

    cfg.acc_banks shouldBe 2
    cfg.acc_singleported shouldBe false
    cfg.acc_sub_banks shouldBe 2

    cfg.acc_bank_entries shouldBe 512

    cfg.spad_read_delay shouldBe 4
    cfg.acc_latency shouldBe 2

    cfg.dma_maxbytes shouldBe 64
    cfg.dma_buswidth shouldBe 128
  }

  it should "enable Sat32 without enabling SCU writeback" in {
    cfg.enablePoTalSat32Acc shouldBe true
    cfg.enablePoTalWriteback shouldBe false

    cfg.mvin_scale_args shouldBe None
    cfg.mvin_scale_acc_args shouldBe None
    cfg.acc_scale_args shouldBe None

    cfg.acc_read_full_width shouldBe true

    cfg.ex_read_from_spad shouldBe true
    cfg.ex_read_from_acc shouldBe false
    cfg.ex_write_to_spad shouldBe false
    cfg.ex_write_to_acc shouldBe true
  }

  it should "reject premature writeback activation" in {
    intercept[IllegalArgumentException] {
      cfg.copy(enablePoTalWriteback = true)
    }
  }

  it should "generate a separate PoTal C header" in {
    cfg.headerFileName shouldBe "gemmini_potal_params.h"

    val header = cfg.generateHeader()

    header should include ("#define DIM 16")
    header should include ("#define BANK_NUM 4")
    header should include ("#define BANK_ROWS 4096")
    header should include ("#define ACC_ROWS 1024")

    header should not include ("#define HAS_MVIN_SCALE")
  }

  it should "leave the default Gemmini configuration unchanged" in {
    val baseline = GemminiConfigs.defaultConfig

    baseline.enablePoTalSat32Acc shouldBe false
    baseline.enablePoTalWriteback shouldBe false

    baseline.dataflow shouldBe Dataflow.BOTH
    baseline.headerFileName shouldBe "gemmini_params.h"
  }
}

