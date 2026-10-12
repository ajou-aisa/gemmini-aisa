package gemmini

import chisel3._
import chisel3.util._

final class PoTalWriteEntry(
  lanes: Int,
  bankWidth: Int,
  rowWidth: Int,
  workWidth: Int,
  robWidth: Int
) extends Bundle {
  val data = Vec(lanes, SInt(32.W))
  val mask = UInt(lanes.W)
  val accBank = UInt(bankWidth.W)
  val accRow = UInt(rowWidth.W)
  val accumulate = Bool()
  val workId = UInt(workWidth.W)
  val robId = UInt(robWidth.W)
  val completesCommand = Bool()
}

// Reserve all output rows before accepting an execution context.
// The ExecuteController cannot stall its already-produced output rows.
final class PoTalWritebackQueue(
  lanes: Int,
  depth: Int,
  bankWidth: Int,
  rowWidth: Int,
  workWidth: Int,
  robWidth: Int
) extends Module {
  require(lanes > 0 && depth >= lanes)

  private val countWidth = math.max(1, log2Ceil(depth + 1))
  private val entryType = new PoTalWriteEntry(
    lanes, bankWidth, rowWidth, workWidth, robWidth)

  val io = IO(new Bundle {
    val reserveRows = Flipped(Decoupled(UInt(countWidth.W)))
    val enq = Flipped(Decoupled(entryType))
    val deq = Decoupled(entryType)
    val reserved = Output(UInt(countWidth.W))
    val queued = Output(UInt(countWidth.W))
    val available = Output(UInt(countWidth.W))
    val missingReservation = Output(Bool())
  })

  private val fifo = Module(new Queue(entryType, depth))
  private val reserved = RegInit(0.U(countWidth.W))
  private val used = fifo.io.count +& reserved
  private val available = depth.U(countWidth.W) - used

  io.reserveRows.ready :=
    io.reserveRows.bits =/= 0.U && io.reserveRows.bits <= available

  // Raw rows can only consume a previously allocated credit.
  fifo.io.enq.valid := io.enq.valid && reserved =/= 0.U
  fifo.io.enq.bits := io.enq.bits
  io.enq.ready := fifo.io.enq.ready && reserved =/= 0.U
  io.deq <> fifo.io.deq

  when(io.reserveRows.fire || io.enq.fire) {
    reserved := reserved +
      Mux(io.reserveRows.fire, io.reserveRows.bits, 0.U) -
      io.enq.fire.asUInt
  }

  io.reserved := reserved
  io.queued := fifo.io.count
  io.available := available
  io.missingReservation := io.enq.valid && reserved === 0.U

  assert(used <= depth.U)
  assert(!(io.enq.valid && reserved =/= 0.U && !fifo.io.enq.ready),
    "PoTal: reserved raw output has no queue space")
}

