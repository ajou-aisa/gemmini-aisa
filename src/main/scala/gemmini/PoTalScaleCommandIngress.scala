
package gemmini

import chisel3._
import chisel3.util._

import GemminiISA._

// ============================================================
// PoTal control-command status
//
// Status is returned to the CPU through RoCCResponse.data
// when Controller integration is implemented.
// ============================================================

object PoTalScaleCommandStatus {
  val OK               = 0.U(8.W)
  val INVALID_COLUMN   = 1.U(8.W)
  val INVALID_SCALE    = 2.U(8.W)
  val SCALE_CONFLICT   = 3.U(8.W)
  val RELEASE_MISS     = 4.U(8.W)
  val INVALID_ADDRESS  = 5.U(8.W)
  val INVALID_ENCODING = 6.U(8.W)
  val BAD_COMMAND      = 7.U(8.W)
}

// Rocket-independent command bundle for unit testing.
// Controller will translate RoCCCommand into this bundle.
final class PoTalScaleCommand extends Bundle {
  val funct = UInt(7.W)
  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
  val rd = UInt(5.W)
  val xd = Bool()
}

final class PoTalScaleCommandResponse extends Bundle {
  val rd = UInt(5.W)
  val status = UInt(8.W)
}

// ============================================================
// Scale command ingress
//
// This module does not own ScaleMemory.
// It talks to the existing PoTalWriteback scale ports.
//
// - WRITE: one carrier to one column/slot
// - RELEASE: generation-qualified entry release
// - FENCE: wait for an external drain condition
//
// Exactly one CPU response may be pending.
// ============================================================

final class PoTalScaleCommandIngress(
  dim: Int,
  scaleEntries: Int = PoTalProtocol.ScaleEntries,
  generationWidth: Int = PoTalProtocol.GenerationWidth
) extends Module {

  require(Set(16, 32, 64).contains(dim))
  require(
    scaleEntries >= 2 &&
    isPow2(scaleEntries) &&
    scaleEntries <= 256
  )
  require(generationWidth == 8)

  private val columnWidth = log2Ceil(dim)
  private val slotWidth = log2Ceil(scaleEntries)

  val io = IO(new Bundle {

    val cmd = Flipped(
      Decoupled(new PoTalScaleCommand)
    )

    val resp = Decoupled(
      new PoTalScaleCommandResponse
    )

    val scaleLoad = Decoupled(
      new PoTalScaleLoad(
        dim, scaleEntries, generationWidth
      )
    )

    val scaleRelease = Decoupled(
      new PoTalScaleRelease(
        dim, scaleEntries, generationWidth
      )
    )

    // From PoTalWriteback diagnostics.
    val scaleConflict = Input(Bool())
    val releaseMiss = Input(Bool())

    // From Controller integration.
    // Both must exclude the current control command
    // itself from the drain condition.
    val releaseSafe = Input(Bool())
    val fenceDrained = Input(Bool())
  })

  // ----------------------------------------------------------
  // Command decoding
  // ----------------------------------------------------------

  val req = io.cmd.bits

  val isWrite =
    req.funct === POTAL_SCALE_WRITE

  val isRelease =
    req.funct === POTAL_SCALE_RELEASE

  val isFence =
    req.funct === POTAL_FENCE

  val address = req.rs1(7, 0)
  val generation = req.rs1(15, 8)
  val column = req.rs1(21, 16)
  val carrier = req.rs2(31, 0)

  val columnValid = column < dim.U
  val addressValid = address < scaleEntries.U
  val carrierValid =
    PoTalScaleEncoding.isValid(carrier)

  val writeEncodingValid =
    !req.rs1(63, 22).orR &&
    !req.rs2(63, 32).orR

  val releaseEncodingValid =
    !req.rs1(63, 22).orR &&
    req.rs2 === 0.U

  val fenceEncodingValid =
    req.rs1 === 0.U &&
    req.rs2 === 0.U

  val writeLegal =
    req.xd &&
    writeEncodingValid &&
    columnValid &&
    addressValid &&
    carrierValid

  val releaseLegal =
    req.xd &&
    releaseEncodingValid &&
    columnValid &&
    addressValid

  // ----------------------------------------------------------
  // Registered CPU response
  // ----------------------------------------------------------

  val responsePending = RegInit(false.B)
  val responseReg =
    Reg(new PoTalScaleCommandResponse)

  io.resp.valid := responsePending
  io.resp.bits := responseReg

  // ----------------------------------------------------------
  // ScaleMemory interfaces
  // ----------------------------------------------------------

  io.scaleLoad.valid :=
    io.cmd.valid &&
    !responsePending &&
    isWrite &&
    writeLegal

  io.scaleLoad.bits.column :=
    column(columnWidth - 1, 0)

  io.scaleLoad.bits.address :=
    address(slotWidth - 1, 0)

  io.scaleLoad.bits.generation :=
    generation

  io.scaleLoad.bits.carrier :=
    carrier

  // RELEASE may be issued only after prior
  // work has drained safely.
  io.scaleRelease.valid :=
    io.cmd.valid &&
    !responsePending &&
    isRelease &&
    releaseLegal &&
    io.releaseSafe

  io.scaleRelease.bits.column :=
    column(columnWidth - 1, 0)

  io.scaleRelease.bits.address :=
    address(slotWidth - 1, 0)

  io.scaleRelease.bits.generation :=
    generation

  // ----------------------------------------------------------
  // Command admission and status generation
  // ----------------------------------------------------------

  io.cmd.ready := false.B

  val nextStatus =
    WireDefault(PoTalScaleCommandStatus.OK)

  when(!responsePending) {

    when(isWrite) {

      when(!req.xd || !writeEncodingValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_ENCODING

      }.elsewhen(!columnValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_COLUMN

      }.elsewhen(!addressValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_ADDRESS

      }.elsewhen(!carrierValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_SCALE

      }.otherwise {
        // ScaleMemory.ready is false when the
        // slot is already allocated.
        //
        // A duplicate write must produce an error,
        // not an infinite stall.
        io.cmd.ready :=
          io.scaleLoad.ready ||
          io.scaleConflict

        when(io.scaleConflict) {
          nextStatus :=
            PoTalScaleCommandStatus.SCALE_CONFLICT
        }
      }

    }.elsewhen(isRelease) {

      when(!req.xd || !releaseEncodingValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_ENCODING

      }.elsewhen(!columnValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_COLUMN

      }.elsewhen(!addressValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_ADDRESS

      }.otherwise {
        io.cmd.ready :=
          io.releaseSafe &&
          io.scaleRelease.ready

        // releaseMiss is combinational and is
        // observed in the release handshake cycle.
        when(io.releaseMiss) {
          nextStatus :=
            PoTalScaleCommandStatus.RELEASE_MISS
        }
      }

    }.elsewhen(isFence) {

      when(!req.xd || !fenceEncodingValid) {
        io.cmd.ready := true.B
        nextStatus :=
          PoTalScaleCommandStatus.INVALID_ENCODING

      }.otherwise {
        io.cmd.ready := io.fenceDrained
      }

    }.otherwise {
      io.cmd.ready := true.B
      nextStatus :=
        PoTalScaleCommandStatus.BAD_COMMAND
    }
  }

  // ----------------------------------------------------------
  // Response lifetime
  // ----------------------------------------------------------

  when(io.cmd.fire) {
    responsePending := true.B
    responseReg.rd := req.rd
    responseReg.status := nextStatus

  }.elsewhen(io.resp.fire) {
    responsePending := false.B
  }

  assert(
    !(io.scaleLoad.fire && io.scaleRelease.fire),
    "PoTal: simultaneous scale load and release"
  )
}

