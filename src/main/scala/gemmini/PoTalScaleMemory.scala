
package gemmini

import chisel3._
import chisel3.util._

// Decoded PoT scale metadata.
final class PoTalScaleMetadata(generationWidth: Int)
    extends Bundle {
  val shift = UInt(15.W)
  val zero = Bool()
  val generation = UInt(generationWidth.W)
}

// Address + generation identify an allocated scale entry.
final class PoTalScaleKey(
  addressWidth: Int,
  generationWidth: Int
) extends Bundle {
  val address = UInt(addressWidth.W)
  val generation = UInt(generationWidth.W)
}

// Scale allocation request.
final class PoTalScaleWrite(
  addressWidth: Int,
  generationWidth: Int
) extends Bundle {
  val address = UInt(addressWidth.W)
  val generation = UInt(generationWidth.W)
  val carrier = UInt(32.W)
}

final class PoTalScaleMemory(
  entries: Int,
  generationWidth: Int
) extends Module {

  require(
    entries > 0 && isPow2(entries),
    "PoTalScaleMemory entries must be a positive power of two"
  )

  require(
    generationWidth > 0,
    "PoTalScaleMemory generationWidth must be positive"
  )

  private val addressWidth =
    math.max(1, log2Ceil(entries))

  val io = IO(new Bundle {

    // Allocate a scale entry.
    val write = Flipped(Decoupled(
      new PoTalScaleWrite(addressWidth, generationWidth)
    ))

    // Release an entry after its ownership lifetime ends.
    val release = Flipped(Valid(
      new PoTalScaleKey(addressWidth, generationWidth)
    ))

    // Combinational metadata lookup.
    val lookup = Input(
      new PoTalScaleKey(addressWidth, generationWidth)
    )

    val hit = Output(Bool())
    val metadata =
      Output(new PoTalScaleMetadata(generationWidth))
    val carrier = Output(UInt(32.W))

    // Protocol diagnostics.
    val invalidWrite = Output(Bool())
    val writeConflict = Output(Bool())
    val releaseMiss = Output(Bool())
  })

  // ----------------------------------------------------------
  // Scale entry storage
  // ----------------------------------------------------------

  private val valid =
    RegInit(VecInit(Seq.fill(entries)(false.B)))

  private val shifts =
    Reg(Vec(entries, UInt(15.W)))

  private val zeroes =
    Reg(Vec(entries, Bool()))

  private val generations =
    Reg(Vec(entries, UInt(generationWidth.W)))

  // ----------------------------------------------------------
  // Write admission
  // ----------------------------------------------------------

  private val writeAddress = io.write.bits.address

  private val writeFree =
    !valid(writeAddress)

  private val writeCarrierValid =
    PoTalScaleEncoding.isValid(io.write.bits.carrier)

  io.write.ready :=
    writeFree && writeCarrierValid

  io.invalidWrite :=
    io.write.valid && !writeCarrierValid

  io.writeConflict :=
    io.write.valid && !writeFree

  // ----------------------------------------------------------
  // Generation-qualified lookup
  // ----------------------------------------------------------

  private val lookupAddress =
    io.lookup.address

  private val lookupHit =
    valid(lookupAddress) &&
    generations(lookupAddress) === io.lookup.generation

  io.hit := lookupHit

  io.metadata.shift := Mux(
    lookupHit,
    shifts(lookupAddress),
    0.U
  )

  io.metadata.zero :=
    lookupHit && zeroes(lookupAddress)

  io.metadata.generation := Mux(
    lookupHit,
    generations(lookupAddress),
    0.U
  )

  // Restore the original 32-bit carrier representation.
  io.carrier := Mux(
    lookupHit,
    Mux(
      zeroes(lookupAddress),
      PoTalScaleEncoding.ZeroCarrier.U(32.W),
      shifts(lookupAddress)
    ),
    0.U(32.W)
  )

  // ----------------------------------------------------------
  // Generation-qualified release
  // ----------------------------------------------------------

  private val releaseAddress =
    io.release.bits.address

  private val releaseHit =
    valid(releaseAddress) &&
    generations(releaseAddress) === io.release.bits.generation

  io.releaseMiss :=
    io.release.valid && !releaseHit

  // ----------------------------------------------------------
  // State updates
  // ----------------------------------------------------------

  when(io.write.fire) {
    valid(writeAddress) := true.B

    shifts(writeAddress) :=
      io.write.bits.carrier(14, 0)

    zeroes(writeAddress) :=
      PoTalScaleEncoding.isZero(io.write.bits.carrier)

    generations(writeAddress) :=
      io.write.bits.generation
  }

  when(io.release.valid && releaseHit) {
    valid(releaseAddress) := false.B
  }
}

