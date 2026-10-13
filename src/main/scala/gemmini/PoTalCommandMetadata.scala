
package gemmini

import chisel3._


object PoTalProtocol {
  // No fixed DIM or scale BlockSize.
  // scaleBlockK = config.DIM * passesPerBlock

  val ScaleEntries = 256
  val GenerationWidth = 8

  val WorkEntries = 128
  val WorkIdWidth = 7
  val WorkIdsPerLoop = 64

  // CPU-side contract:
  // passesPerBlock = scaleBlockK / DIM
  // passesPerBlock >= 1
  // scaleBlockK % DIM == 0

  def passesPerBlock(
    dim: Int,
    scaleBlockK: Int
  ): Int = {
    require(dim > 0)
    require(scaleBlockK >= dim)
    require(scaleBlockK % dim == 0)

    val passes = scaleBlockK / dim

    require(passes <= 65535)
    passes
  }
}


class PoTalCommandMetadata extends Bundle {
  val valid = Bool()

  // Scale block index within the active scale window.
  // One block contains passesPerBlock array K tiles.
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

