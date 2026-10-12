
package gemmini

import chisel3._
import chisel3.iotesters._
import Arithmetic.SIntArithmetic

object PoTalAdderGolden {
  val max32: BigInt = (BigInt(1) << 31) - 1
  val min32: BigInt = -(BigInt(1) << 31)
  val mask32: BigInt = (BigInt(1) << 32) - 1

  def add(a: BigInt, b: BigInt, saturate: Boolean): BigInt = {
    val sum = a + b

    if (saturate) {
      sum.max(min32).min(max32)
    } else {
      val bits = sum & mask32
      if (bits.testBit(31)) bits - (BigInt(1) << 32)
      else bits
    }
  }
}


// ------------------------------------------------------------
// Test 1: Individual AccPipe
// ------------------------------------------------------------

class PoTalAccPipeTester(
  dut: AccPipe[SInt],
  latency: Int,
  saturate: Boolean
) extends PeekPokeTester(dut) {

  import PoTalAdderGolden._

  val samples: Seq[(BigInt, BigInt)] = Seq(
    (max32, BigInt(1)),
    (min32, BigInt(-1)),
    (max32, min32),
    (BigInt(100), BigInt(200)),
    (BigInt(-100), BigInt(40)),
    (max32, BigInt(10)),
    (min32, BigInt(-10)),
    (BigInt(0), BigInt(0)),
    (BigInt(1000000), BigInt(-1000000))
  )

  // One input vector per cycle.
  // After latency cycles, compare the corresponding output.
  for (cycle <- 0 until samples.length + latency - 1) {
    val (a, b) =
      if (cycle < samples.length) samples(cycle)
      else (BigInt(0), BigInt(0))

    poke(dut.io.op1, a)
    poke(dut.io.op2, b)

    step(1)

    if (cycle >= latency - 1) {
      val (ea, eb) = samples(cycle - latency + 1)
      expect(dut.io.sum, add(ea, eb, saturate))
    }
  }
}


// ------------------------------------------------------------
// Test 2: AccPipeShared (2 banks, 2x2 output lanes)
// ------------------------------------------------------------

class PoTalAccPipeSharedTester(
  dut: AccPipeShared[SInt],
  latency: Int,
  saturate: Boolean
) extends PeekPokeTester(dut) {

  import PoTalAdderGolden._

  type Operands = Seq[(BigInt, BigInt)]

  val samples: Seq[(Int, Operands)] = Seq(
    (
      0,
      Seq(
        (max32, BigInt(1)),
        (min32, BigInt(-1)),
        (BigInt(100), BigInt(200)),
        (BigInt(-100), BigInt(40))
      )
    ),
    (
      1,
      Seq(
        (max32, min32),
        (max32, BigInt(2)),
        (min32, BigInt(-2)),
        (BigInt(7), BigInt(-3))
      )
    ),
    (
      0,
      Seq(
        (BigInt(0), BigInt(0)),
        (BigInt(-1), BigInt(-1)),
        (BigInt(55), BigInt(45)),
        (max32, max32)
      )
    ),
    (
      1,
      Seq(
        (min32, min32),
        (BigInt(1234), BigInt(-1234)),
        (BigInt(-99), BigInt(100)),
        (BigInt(42), BigInt(10))
      )
    )
  )

  for (cycle <- 0 until samples.length + latency - 1) {
    val (selectedBank, operands) =
      if (cycle < samples.length) {
        samples(cycle)
      } else {
        (0, Seq.fill(4)((BigInt(0), BigInt(0))))
      }

    for (bank <- 0 until 2) {
      poke(dut.io.in_sel(bank), if (bank == selectedBank) 1 else 0)

      for (row <- 0 until 2; col <- 0 until 2) {
        val index = row * 2 + col

        val (a, b) =
          if (bank == selectedBank) operands(index)
          else (
            BigInt(10000 + bank * 100 + index),
            BigInt(-5000 - bank * 100 - index)
          )

        poke(dut.io.ina(bank)(row)(col), a)
        poke(dut.io.inb(bank)(row)(col), b)
      }
    }

    step(1)

    if (cycle >= latency - 1) {
      val (_, expectedOperands) = samples(cycle - latency + 1)

      for (row <- 0 until 2; col <- 0 until 2) {
        val index = row * 2 + col
        val (ea, eb) = expectedOperands(index)

        expect(
          dut.io.out(row)(col),
          add(ea, eb, saturate)
        )
      }
    }
  }
}


// ------------------------------------------------------------
// ScalaTest
// ------------------------------------------------------------

class PoTalAccPipeSpec extends ChiselFlatSpec {

  behavior of "PoTal Accumulator Adder"

  for {
    saturate <- Seq(false, true)
    latency <- Seq(1, 2)
  } {
    val mode = if (saturate) "sat32" else "wrap32"

    it should s"compute AccPipe $mode at latency $latency" in {
      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-accpipe-$mode-$latency"
        ),
        () => new AccPipe(
          latency = latency,
          t = SInt(32.W),
          saturateInt32 = saturate
        )
      ) { dut =>
        new PoTalAccPipeTester(dut, latency, saturate)
      } should be (true)
    }

    it should s"share AccPipe across banks in $mode at latency $latency" in {
      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-accpipe-shared-$mode-$latency"
        ),
        () => new AccPipeShared(
          latency = latency,
          t = Vec(2, Vec(2, SInt(32.W))),
          banks = 2,
          saturateInt32 = saturate
        )
      ) { dut =>
        new PoTalAccPipeSharedTester(dut, latency, saturate)
      } should be (true)
    }
  }
}

