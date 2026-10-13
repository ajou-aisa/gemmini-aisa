
package gemmini

import chisel3._
import chisel3.iotesters._

class PoTalScaleBlockCounterTester(
  dut: PoTalScaleBlockCounter,
  passes: Int,
  kTiles: Int
) extends PeekPokeTester(dut) {

  poke(dut.io.start, 0)
  poke(dut.io.passesPerBlock, passes)
  poke(dut.io.commandAccepted, 0)
  poke(dut.io.isCompute, 0)
  poke(dut.io.lastOutputTile, 0)

  reset(3)

  // Begin a new matmul.
  poke(dut.io.start, 1)
  step(1)
  poke(dut.io.start, 0)

  val outputTiles = 4 // maxI=2, maxJ=2

  for (k <- 0 until kTiles) {
    val expectedBlock = k / passes
    val expectedPass = k % passes

    expect(dut.io.blockIndex, expectedBlock)
    expect(dut.io.passInBlock, expectedPass)

    for (tile <- 0 until outputTiles) {
      val isLastTile = tile == outputTiles - 1

      poke(dut.io.lastOutputTile,
        if (isLastTile) 1 else 0)

      // PRELOAD accepted: no scale progress.
      poke(dut.io.isCompute, 0)
      poke(dut.io.commandAccepted, 1)
      step(1)

      expect(dut.io.blockIndex, expectedBlock)
      expect(dut.io.passInBlock, expectedPass)

      // COMPUTE stalled: no scale progress.
      poke(dut.io.isCompute, 1)
      poke(dut.io.commandAccepted, 0)

      for (_ <- 0 until 3) {
        expect(dut.io.blockIndex, expectedBlock)
        expect(dut.io.passInBlock, expectedPass)
        step(1)
      }

      // COMPUTE accepted.
      poke(dut.io.commandAccepted, 1)
      step(1)

      if (!isLastTile) {
        expect(dut.io.blockIndex, expectedBlock)
        expect(dut.io.passInBlock, expectedPass)
      }
    }
  }

  // All K tile groups completed.
  expect(dut.io.blockIndex, kTiles / passes)
  expect(dut.io.passInBlock, kTiles % passes)

  // A new matmul must reset both counters.
  poke(dut.io.commandAccepted, 0)
  poke(dut.io.isCompute, 0)
  poke(dut.io.start, 1)

  step(1)

  poke(dut.io.start, 0)
  expect(dut.io.blockIndex, 0)
  expect(dut.io.passInBlock, 0)
}

class PoTalScaleBlockSpec extends ChiselFlatSpec {

  behavior of "PoTal variable scale block scheduling"

  val cases = Seq(
    (16, 32, 5),
    (32, 96, 7),
    (64, 64, 3),
    (64, 128, 5)
  )

  for ((dim, blockSize, kTiles) <- cases) {
    it should s"schedule DIM=$dim BlockSize=$blockSize" in {
      val passes =
        PoTalProtocol.passesPerBlock(dim, blockSize)

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-scale-$dim-$blockSize"
        ),
        () => new PoTalScaleBlockCounter(16)
      ) { dut =>
        new PoTalScaleBlockCounterTester(
          dut,
          passes,
          kTiles
        )
      } should be (true)
    }
  }

  it should "reject misaligned scale block sizes" in {
    assert(PoTalProtocol.passesPerBlock(16, 32) == 2)
    assert(PoTalProtocol.passesPerBlock(32, 96) == 3)
    assert(PoTalProtocol.passesPerBlock(64, 64) == 1)
    assert(PoTalProtocol.passesPerBlock(64, 128) == 2)

    intercept[IllegalArgumentException] {
      PoTalProtocol.passesPerBlock(64, 32)
    }

    intercept[IllegalArgumentException] {
      PoTalProtocol.passesPerBlock(64, 96)
    }
  }
}

