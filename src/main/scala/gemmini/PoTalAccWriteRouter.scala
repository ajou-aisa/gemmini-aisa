
package gemmini

import chisel3._
import chisel3.util._
import Arithmetic.SIntArithmetic

// Routes EX raw ACC writes to either:
//
// 1. Conventional ACC path (unscaled)
// 2. PoTalWriteback raw input (SCU path)
//
// PoTal scaled writes share the physical ACC port.
// Conventional EX writes have priority because
// the original ExecuteController cannot stall outputs.
final class PoTalAccWriteRouter(
  dim: Int,
  accBanks: Int,
  accBankEntries: Int
) extends Module {

  require(Set(16, 32, 64).contains(dim))
  require(accBanks >= 2)
  require(accBankEntries >= 2)

  private def writeType =
    new AccumulatorWriteReq(
      accBankEntries,
      Vec(dim, Vec(1, SInt(32.W)))
    )

  val io = IO(new Bundle {

    // From ExecuteController.
    val exWrite = Flipped(
      Vec(accBanks, Decoupled(writeType))
    )

    // Synchronized with EX raw output.
    val isPoTal = Input(Vec(accBanks, Bool()))

    // To PoTalWriteback.rawWrite.
    val rawWrite =
      Vec(accBanks, Decoupled(writeType))

    // From PoTalWriteback.scaledWrite.
    val scaledWrite = Flipped(
      Vec(accBanks, Decoupled(writeType))
    )

    // To physical ACC SRAM.
    val accWrite =
      Vec(accBanks, Decoupled(writeType))
  })

  for (bank <- 0 until accBanks) {

    val ex = io.exWrite(bank)
    val raw = io.rawWrite(bank)

    // Fixed-priority arbiter:
    // input 0 = conventional EX
    // input 1 = PoTal scaled write
    val arb = Module(
      new Arbiter(writeType, 2)
    )

    // --------------------------------------------
    // PoTal raw output bypasses physical ACC.
    // --------------------------------------------

    raw.valid :=
      ex.valid && io.isPoTal(bank)

    raw.bits := ex.bits

    // --------------------------------------------
    // Conventional output goes directly to ACC.
    // --------------------------------------------

    arb.io.in(0).valid :=
      ex.valid && !io.isPoTal(bank)

    arb.io.in(0).bits := ex.bits

    // --------------------------------------------
    // Scaled PoTal output also targets ACC.
    // --------------------------------------------

    arb.io.in(1) <> io.scaledWrite(bank)

    // Physical accumulator destination.
    io.accWrite(bank) <> arb.io.out

    // EX handshakes with its selected destination.
    ex.ready := Mux(
      io.isPoTal(bank),
      raw.ready,
      arb.io.in(0).ready
    )
  }
}

