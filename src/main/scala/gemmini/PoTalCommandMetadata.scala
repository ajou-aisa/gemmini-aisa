
package gemmini

import chisel3._

object PoTalProtocol {
  val Dim = 16
  val KBlockSize = 32

  val ScaleEntries = 256
  val GenerationWidth = 8

  val WorkEntries = 128
  val WorkIdWidth = 7

  // LoopMatmul supports up to two concurrent loops.
  // Reserve 64 work IDs for each loop.
  val WorkIdsPerLoop = 64
}

class PoTalCommandMetadata extends Bundle {
  val valid = Bool()

  // K32 block index within the current scale window.
  val blockIndex = UInt(16.W)

  val scaleBase = UInt(16.W)
  val scaleGeneration =
    UInt(PoTalProtocol.GenerationWidth.W)

  val outputTileCount = UInt(16.W)
  val outputTileIndex = UInt(16.W)

  val workId = UInt(PoTalProtocol.WorkIdWidth.W)

  // These fields control ACC/work semantics,
  // not scale selection.
  val firstContribution = Bool()
  val lastContribution = Bool()
}

