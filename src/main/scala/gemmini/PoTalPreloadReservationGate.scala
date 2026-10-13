
package gemmini

import chisel3._
import chisel3.util._

// Atomically accept a PoTal PRELOAD and reserve
// its output context before execution.
//
// Normal Gemmini commands pass through unchanged.
final class PoTalPreloadReservationGate(
  dim: Int,
  robWidth: Int
) extends Module {

  require(Set(16, 32, 64).contains(dim))
  require(robWidth > 0)

  private val shapeWidth = log2Ceil(dim + 1)

  val io = IO(new Bundle {

    // ReservationStation issue side
    val sourceValid = Input(Bool())
    val sourceReady = Output(Bool())

    val isPreload = Input(Bool())
    val metadata = Input(new PoTalCommandMetadata)

    val robId = Input(UInt(robWidth.W))
    val validRows = Input(UInt(shapeWidth.W))
    val validColumns = Input(UInt(shapeWidth.W))

    // ExecuteController command side
    val executeValid = Output(Bool())
    val executeReady = Input(Bool())

    // PoTalWriteback context reservation
    val contextIssue = Decoupled(
      new PoTalOutputContext(
        dim,
        robWidth,
        PoTalProtocol.WorkIdWidth,
        PoTalProtocol.GenerationWidth
      )
    )
  })

  private val needsContext =
    io.isPreload && io.metadata.valid

  // ----------------------------------------------------------
  // Atomic acceptance
  // ----------------------------------------------------------

  // The ExecuteController must not accept a PoTal
  // PRELOAD unless context reservation is ready.
  io.executeValid :=
    io.sourceValid &&
    (!needsContext || io.contextIssue.ready)

  io.sourceReady :=
    io.executeReady &&
    (!needsContext || io.contextIssue.ready)

  // The context must not be accepted unless the
  // ExecuteController can accept the same command.
  io.contextIssue.valid :=
    io.sourceValid &&
    needsContext &&
    io.executeReady

  // ----------------------------------------------------------
  // Metadata mapping
  // ----------------------------------------------------------

  io.contextIssue.bits.robId :=
    io.robId

  io.contextIssue.bits.workId :=
    io.metadata.workId

  io.contextIssue.bits.blockIndex :=
    io.metadata.blockIndex

  io.contextIssue.bits.scaleBase :=
    io.metadata.scaleBase

  io.contextIssue.bits.scaleGeneration :=
    io.metadata.scaleGeneration

  io.contextIssue.bits.outputTileCount :=
    io.metadata.outputTileCount

  io.contextIssue.bits.outputTileIndex :=
    io.metadata.outputTileIndex

  io.contextIssue.bits.validRows :=
    io.validRows

  io.contextIssue.bits.validColumns :=
    io.validColumns

  io.contextIssue.bits.firstContribution :=
    io.metadata.firstContribution

  io.contextIssue.bits.lastContribution :=
    io.metadata.lastContribution

  // ----------------------------------------------------------
  // Protocol assertions
  // ----------------------------------------------------------

  when(io.sourceValid && io.metadata.valid) {
    assert(
      io.isPreload,
      "PoTal metadata must belong to PRELOAD"
    )
  }

  when(io.contextIssue.fire) {
    assert(
      io.validRows > 0.U &&
      io.validRows <= dim.U,
      "PoTal: invalid output row count"
    )

    assert(
      io.validColumns > 0.U &&
      io.validColumns <= dim.U,
      "PoTal: invalid output column count"
    )
  }

  // No partial acceptance is permitted.
  when(io.sourceValid && needsContext) {
    assert(
      (io.executeValid && io.executeReady) ===
      io.contextIssue.fire,
      "PoTal PRELOAD/context acceptance mismatch"
    )
  }
}

