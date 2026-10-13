
package gemmini

import chisel3._
import chisel3.iotesters._

final class PoTalPreloadReservationGateTester(
  dut: PoTalPreloadReservationGate,
  dim: Int
) extends PeekPokeTester(dut) {

  poke(dut.io.sourceValid, 0)
  poke(dut.io.isPreload, 0)
  poke(dut.io.executeReady, 0)
  poke(dut.io.contextIssue.ready, 0)

  poke(dut.io.robId, 5)
  poke(dut.io.validRows, dim - 1)
  poke(dut.io.validColumns, dim)

  poke(dut.io.metadata.valid, 0)
  poke(dut.io.metadata.workId, 11)
  poke(dut.io.metadata.blockIndex, 3)
  poke(dut.io.metadata.scaleBase, 20)
  poke(dut.io.metadata.scaleGeneration, 7)
  poke(dut.io.metadata.outputTileCount, 4)
  poke(dut.io.metadata.outputTileIndex, 2)
  poke(dut.io.metadata.firstContribution, 1)
  poke(dut.io.metadata.lastContribution, 0)

  reset(2)

  // ----------------------------------------------------------
  // Normal Gemmini command: no context reservation
  // ----------------------------------------------------------

  poke(dut.io.sourceValid, 1)
  poke(dut.io.isPreload, 0)
  poke(dut.io.executeReady, 1)

  expect(dut.io.executeValid, 1)
  expect(dut.io.sourceReady, 1)
  expect(dut.io.contextIssue.valid, 0)

  step(1)

  // ----------------------------------------------------------
  // PoTal PRELOAD, context queue stalled
  // ----------------------------------------------------------

  poke(dut.io.isPreload, 1)
  poke(dut.io.metadata.valid, 1)
  poke(dut.io.contextIssue.ready, 0)

  for (_ <- 0 until 3) {
    expect(dut.io.executeValid, 0)
    expect(dut.io.sourceReady, 0)
    expect(dut.io.contextIssue.valid, 1)
    step(1)
  }

  // ----------------------------------------------------------
  // Context ready, ExecuteController stalled
  // ----------------------------------------------------------

  poke(dut.io.contextIssue.ready, 1)
  poke(dut.io.executeReady, 0)

  expect(dut.io.executeValid, 1)
  expect(dut.io.sourceReady, 0)
  expect(dut.io.contextIssue.valid, 0)

  step(1)

  // ----------------------------------------------------------
  // Both ready: one atomic acceptance
  // ----------------------------------------------------------

  poke(dut.io.executeReady, 1)

  expect(dut.io.executeValid, 1)
  expect(dut.io.sourceReady, 1)
  expect(dut.io.contextIssue.valid, 1)

  expect(dut.io.contextIssue.bits.robId, 5)
  expect(dut.io.contextIssue.bits.workId, 11)
  expect(dut.io.contextIssue.bits.blockIndex, 3)
  expect(dut.io.contextIssue.bits.scaleBase, 20)
  expect(dut.io.contextIssue.bits.scaleGeneration, 7)
  expect(dut.io.contextIssue.bits.outputTileCount, 4)
  expect(dut.io.contextIssue.bits.outputTileIndex, 2)
  expect(dut.io.contextIssue.bits.validRows, dim - 1)
  expect(dut.io.contextIssue.bits.validColumns, dim)
  expect(dut.io.contextIssue.bits.firstContribution, 1)
  expect(dut.io.contextIssue.bits.lastContribution, 0)

  step(1)

  // No source command: no duplicate reservation
  poke(dut.io.sourceValid, 0)

  expect(dut.io.executeValid, 0)
  expect(dut.io.contextIssue.valid, 0)

  step(1)

  // Normal PRELOAD without PoTal metadata
  // must still execute if WB context queue is full.
  poke(dut.io.sourceValid, 1)
  poke(dut.io.metadata.valid, 0)
  poke(dut.io.contextIssue.ready, 0)

  expect(dut.io.executeValid, 1)
  expect(dut.io.sourceReady, 1)
  expect(dut.io.contextIssue.valid, 0)
}

class PoTalPreloadReservationGateSpec
    extends ChiselFlatSpec {

  behavior of "PoTal PRELOAD context reservation"

  for (dim <- Seq(16, 32, 64)) {
    it should s"atomically reserve DIM=$dim contexts" in {

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-preload-gate-d$dim"
        ),
        () => new PoTalPreloadReservationGate(
          dim = dim,
          robWidth = 6
        )
      ) { dut =>
        new PoTalPreloadReservationGateTester(dut, dim)
      } should be (true)
    }
  }
}

