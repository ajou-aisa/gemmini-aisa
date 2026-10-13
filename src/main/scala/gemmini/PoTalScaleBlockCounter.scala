
package gemmini

import chisel3._

final class PoTalScaleBlockCounter(
  width: Int = 16
) extends Module {
  require(width > 0)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val passesPerBlock = Input(UInt(width.W))

    val commandAccepted = Input(Bool())
    val isCompute = Input(Bool())
    val lastOutputTile = Input(Bool())

    val blockIndex = Output(UInt(width.W))
    val passInBlock = Output(UInt(width.W))
  })

  private val pass = RegInit(0.U(width.W))
  private val block = RegInit(0.U(width.W))

  // PRELOAD never advances the K scale position.
  // The final output tile's COMPUTE must be accepted.
  private val advance =
    io.commandAccepted &&
    io.isCompute &&
    io.lastOutputTile

  when(io.start) {
    pass := 0.U
    block := 0.U

  }.elsewhen(advance) {
    assert(
      io.passesPerBlock =/= 0.U,
      "PoTal: passesPerBlock must be positive"
    )

    when(pass === io.passesPerBlock - 1.U) {
      pass := 0.U
      block := block + 1.U
    }.otherwise {
      pass := pass + 1.U
    }
  }

  io.blockIndex := block
  io.passInBlock := pass
}

