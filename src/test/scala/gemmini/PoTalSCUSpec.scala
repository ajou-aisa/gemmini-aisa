
package gemmini

import chisel3._
import chisel3.iotesters._

class PoTalSCUTester(dut: PoTalSCU, partialWidth: Int)
    extends PeekPokeTester(dut) {

  private val int32Max = (BigInt(1) << 31) - 1
  private val int32Min = -(BigInt(1) << 31)

  private val partialMax = (BigInt(1) << (partialWidth - 1)) - 1
  private val partialMin = -(BigInt(1) << (partialWidth - 1))

  private val zeroCarrier = BigInt("80000000", 16)

  // Independent HP1 numerical reference using arbitrary-precision integers.
  // Do not derive expected results from the DUT's shift logic.
  private def golden(
    partial: BigInt,
    carrier: BigInt
  ): (BigInt, Boolean, Boolean) = {

    val zeroScale = carrier == zeroCarrier
    val valid = zeroScale || (carrier >= 0 && carrier <= 32767)

    if (!valid || zeroScale) {
      (BigInt(0), valid, zeroScale)
    } else {
      val shifted =
        if (partial == 0) {
          BigInt(0)
        } else if (carrier >= 32) {
          if (partial > 0) int32Max else int32Min
        } else {
          partial << carrier.toInt
        }

      val saturated = shifted.max(int32Min).min(int32Max)
      (saturated, true, false)
    }
  }

  private def check(partial: BigInt, carrier: BigInt): Unit = {
    assert(partial >= partialMin && partial <= partialMax)
    assert(carrier >= 0 && carrier < (BigInt(1) << 32))

    val (expected, valid, zero) = golden(partial, carrier)

    poke(dut.io.partial, partial)
    poke(dut.io.carrier, carrier)

    step(1)

    expect(dut.io.result, expected)
    expect(dut.io.carrierValid, if (valid) 1 else 0)
    expect(dut.io.zeroScale, if (zero) 1 else 0)
  }

  // ------------------------------------------------------------
  // 1. Original SCU numerical vectors
  // ------------------------------------------------------------

  check(7, 0)
  check(-3, 1)

  check(1, 30)
  check(-1, 31)
  check(1, 31)
  check(-1, 32)

  check(2, 30)
  check(-3, 30)

  check(1, 32767)
  check(0, 32767)

  check(123, zeroCarrier)

  check(1, 32768)
  check(1, BigInt("ffffffff", 16))

  // ------------------------------------------------------------
  // 2. Signed boundaries and saturation
  // ------------------------------------------------------------

  val partialValues = Seq(
    BigInt(0),
    BigInt(1),
    BigInt(-1),
    BigInt(2),
    BigInt(-2),
    BigInt(7),
    BigInt(-7),
    partialMax,
    partialMin
  ).distinct

  val carriers = Seq(
    BigInt(0),
    BigInt(1),
    BigInt(2),
    BigInt(7),
    BigInt(15),
    BigInt(30),
    BigInt(31),
    BigInt(32),
    BigInt(32767),
    BigInt(32768),
    zeroCarrier,
    BigInt("ffffffff", 16)
  )

  for {
    partial <- partialValues
    carrier <- carriers
  } {
    check(partial, carrier)
  }

  // ------------------------------------------------------------
  // 3. Deterministic randomized regression
  // ------------------------------------------------------------

  val random = new scala.util.Random(0x50A1L + partialWidth)

  val widthMask = (BigInt(1) << partialWidth) - 1
  val modulus = BigInt(1) << partialWidth

  for (_ <- 0 until 64) {
    val raw = BigInt(random.nextLong()) & widthMask

    val signed =
      if (raw.testBit(partialWidth - 1)) raw - modulus
      else raw

    val carrier = carriers(random.nextInt(carriers.length))

    check(signed, carrier)
  }
}

class PoTalSCUSpec extends ChiselFlatSpec {

  behavior of "PoTalSCU"

  // A4W4: DIM16/32/64 => 12/13/14 bits
  // A8W8: DIM16/32/64 => 20/21/22 bits
  val partialWidths = Seq(12, 13, 14, 20, 21, 22)

  for (width <- partialWidths) {
    it should s"match the HP1 numerical oracle at width $width" in {

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir", s"test_run_dir/potal-scu-$width"
        ),
        () => new PoTalSCU(width)
      ) { dut =>
        new PoTalSCUTester(dut, width)
      } should be (true)
    }
  }
}

