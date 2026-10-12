package gemmini

import chisel3._

object PoTalScaleEncoding {
  val ZeroCarrier: BigInt = BigInt("80000000", 16)
  val MaxShift: Int = 32767

  def isZero(carrier: UInt): Bool =
    carrier === ZeroCarrier.U(32.W)

  def isShift(carrier: UInt): Bool =
    carrier <= MaxShift.U(32.W)

  def isValid(carrier: UInt): Bool =
    isZero(carrier) || isShift(carrier)
}

class PoTalSCU(partialWidth: Int) extends Module {
  require(partialWidth >= 2 && partialWidth <= 32)

  val io = IO(new Bundle {
    val partial = Input(SInt(partialWidth.W))
    val carrier = Input(UInt(32.W))

    val result = Output(SInt(32.W))
    val carrierValid = Output(Bool())
    val zeroScale = Output(Bool())
  })

  val zeroScale = PoTalScaleEncoding.isZero(io.carrier)
  val shiftScale = PoTalScaleEncoding.isShift(io.carrier)

  val shiftAmount = io.carrier(4, 0)
  val partial = io.partial.pad(33)

  val positiveLimit = 2147483647L.S(33.W) >> shiftAmount
  val negativeLimit = (-2147483648L).S(33.W) >> shiftAmount

  val shifted = Wire(SInt(64.W))
  shifted := partial << shiftAmount

  io.carrierValid := zeroScale || shiftScale
  io.zeroScale := zeroScale
  io.result := 0.S(32.W)

  when(shiftScale && io.partial =/= 0.S) {
    when(io.carrier >= 32.U) {
      io.result := Mux(
        io.partial < 0.S,
        (-2147483648L).S,
        2147483647L.S
      )
    }.elsewhen(partial > positiveLimit) {
      io.result := 2147483647L.S
    }.elsewhen(partial < negativeLimit) {
      io.result := (-2147483648L).S
    }.otherwise {
      io.result := shifted.asUInt(31, 0).asSInt
    }
  }
}
