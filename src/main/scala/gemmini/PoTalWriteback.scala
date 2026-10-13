package gemmini

import chisel3._
import chisel3.util._
import Arithmetic.SIntArithmetic

// A scale block and output N tile map to a scale slot:
//   scaleBase + blockIndex * outputTileCount + outputTileIndex.
//
// Each scale block contains passesPerBlock physical K tiles.
final class PoTalOutputContext(
  dim: Int,
  robWidth: Int,
  workWidth: Int,
  generationWidth: Int
) extends Bundle {
  val robId = UInt(robWidth.W)
  val workId = UInt(workWidth.W)
  val blockIndex = UInt(16.W)
  val scaleBase = UInt(16.W)
  val outputTileCount = UInt(16.W)
  val outputTileIndex = UInt(16.W)
  val scaleGeneration = UInt(generationWidth.W)
  val validRows = UInt(log2Ceil(dim + 1).W)
  val validColumns = UInt(log2Ceil(dim + 1).W)
  val firstContribution = Bool()
  val lastContribution = Bool()
}

final class PoTalScaleLoad(
  dim: Int,
  entries: Int,
  generationWidth: Int
) extends Bundle {
  val column = UInt(math.max(1, log2Ceil(dim)).W)
  val address = UInt(math.max(1, log2Ceil(entries)).W)
  val generation = UInt(generationWidth.W)
  val carrier = UInt(32.W)
}

final class PoTalScaleRelease(
  dim: Int,
  entries: Int,
  generationWidth: Int
) extends Bundle {
  val column = UInt(math.max(1, log2Ceil(dim)).W)
  val address = UInt(math.max(1, log2Ceil(entries)).W)
  val generation = UInt(generationWidth.W)
}

final class PoTalQueuedContext(
  dim: Int,
  robWidth: Int,
  workWidth: Int,
  generationWidth: Int,
  slotWidth: Int
) extends Bundle {
  val context = new PoTalOutputContext(
    dim, robWidth, workWidth, generationWidth)
  val scaleSlot = UInt(slotWidth.W)
}

// Standalone PoTal writeback. The Controller and RoCC command/scale-loading
// ingress are deliberately NOT connected in this module.
final class PoTalWriteback[T <: Data : Arithmetic, U <: Data, V <: Data](
  config: GemminiArrayConfig[T, U, V],
  scaleEntries: Int = 256,
  generationWidth: Int = 8,
  workEntries: Int = 128,
  contextDepth: Int = 2,
  queueDepth: Int = -1
) extends Module {
  private val dim = config.DIM
  private val depth = if (queueDepth < 0) 2 * dim else queueDepth
  require(config.dataflow == Dataflow.WS)
  require(config.inputType.isInstanceOf[SInt])
  require(config.spatialArrayOutputType.isInstanceOf[SInt])
  require(config.accType.isInstanceOf[SInt])
  require(config.tileColumns == 1 && config.tileRows == 1)
  require(config.accType.getWidth == 32)
  require(config.acc_banks >= 2)
  require(config.acc_latency >= 2)
  require(config.spatialArrayOutputType.getWidth >= 2 &&
    config.spatialArrayOutputType.getWidth <= 32)
  require(scaleEntries >= 2 && isPow2(scaleEntries))
  require(workEntries >= 2 && isPow2(workEntries))
  require(contextDepth >= 1 && depth >= dim)

  private val slotWidth = math.max(1, log2Ceil(scaleEntries))
  private val workWidth = math.max(1, log2Ceil(workEntries))
  private val bankWidth = math.max(1, log2Ceil(config.acc_banks))
  private val rowWidth = math.max(1, log2Ceil(config.acc_bank_entries))
  private val robWidth = config.ROB_ID_WIDTH
  private val robEntries = 1 << robWidth
  private val rawBits = config.spatialArrayOutputType.getWidth
  private def writeType = new AccumulatorWriteReq(
    config.acc_bank_entries,
    Vec(config.meshColumns, Vec(config.tileColumns, config.accType)))

  val io = IO(new Bundle {
    val contextIssue = Flipped(Decoupled(new PoTalOutputContext(
      dim, robWidth, workWidth, generationWidth)))
    val rawWrite = Flipped(Vec(config.acc_banks, Decoupled(writeType)))
    val rawCompletion = Flipped(Valid(UInt(robWidth.W)))
    val scaledWrite = Vec(config.acc_banks, Decoupled(writeType))
    val scaleLoad = Flipped(Decoupled(new PoTalScaleLoad(
      dim, scaleEntries, generationWidth)))
    val scaleRelease = Flipped(Decoupled(new PoTalScaleRelease(
      dim, scaleEntries, generationWidth)))
    val robCompletion = Decoupled(UInt(robWidth.W))
    val workDone = Decoupled(UInt(workWidth.W))
    val pipelineDrained = Output(Bool())
    val drained = Output(Bool())
    val error = Output(new Bundle {
      val scaleMiss = Bool()
      val invalidScale = Bool()
      val scaleConflict = Bool()
      val releaseMiss = Bool()
      val rawWithoutContext = Bool()
      val reservationError = Bool()
      val completionOverflow = Bool()
      val workProtocol = Bool()
    })
  })

  private val contexts = Module(new Queue(new PoTalQueuedContext(
    dim, robWidth, workWidth, generationWidth, slotWidth), contextDepth))
  private val writes = Module(new PoTalWritebackQueue(
    dim, depth, bankWidth, rowWidth, workWidth, robWidth))
  private val scales = Seq.fill(dim)(Module(new PoTalScaleMemory(
    scaleEntries, generationWidth)))
  private val tracker = Module(new PoTalWorkTracker(workEntries))

  private val pendingRob = RegInit(VecInit(Seq.fill(robEntries)(false.B)))
  private val rawDone = RegInit(VecInit(Seq.fill(robEntries)(false.B)))
  private val writeDone = RegInit(VecInit(Seq.fill(robEntries)(false.B)))

  // Scale cache: one independent carrier per output column.
  io.scaleLoad.ready := VecInit(scales.map(_.io.write.ready))(
    io.scaleLoad.bits.column)
  io.scaleRelease.ready := contexts.io.count === 0.U &&
    !io.contextIssue.valid && !io.rawWrite.map(_.valid).reduce(_ || _)

  private val rawValid = io.rawWrite.map(_.valid).reduce(_ || _)
  private val rawBank = PriorityEncoder(VecInit(io.rawWrite.map(_.valid)))
  private val raw = io.rawWrite(rawBank).bits
  private val head = contexts.io.deq.bits.context
  private val incoming = io.contextIssue.bits

  private val slotWide = incoming.scaleBase +&
    (incoming.blockIndex * incoming.outputTileCount) +&
    incoming.outputTileIndex
  private val shapeValid = incoming.validRows > 0.U &&
    incoming.validRows <= dim.U &&
    incoming.validColumns > 0.U &&
    incoming.validColumns <= dim.U &&
    incoming.outputTileCount > 0.U &&
    incoming.outputTileIndex < incoming.outputTileCount &&
    slotWide < scaleEntries.U

  private val lookupSlot = Mux(rawValid,
    contexts.io.deq.bits.scaleSlot, slotWide(slotWidth - 1, 0))
  private val lookupGeneration = Mux(rawValid,
    head.scaleGeneration, incoming.scaleGeneration)
  private val activeColumns = Mux(rawValid,
    head.validColumns, incoming.validColumns)

  for ((memory, column) <- scales.zipWithIndex) {
    memory.io.write.valid := io.scaleLoad.valid &&
      io.scaleLoad.bits.column === column.U
    memory.io.write.bits.address := io.scaleLoad.bits.address
    memory.io.write.bits.generation := io.scaleLoad.bits.generation
    memory.io.write.bits.carrier := io.scaleLoad.bits.carrier

    memory.io.release.valid := io.scaleRelease.fire &&
      io.scaleRelease.bits.column === column.U
    memory.io.release.bits.address := io.scaleRelease.bits.address
    memory.io.release.bits.generation := io.scaleRelease.bits.generation
    memory.io.lookup.address := lookupSlot
    memory.io.lookup.generation := lookupGeneration
  }
  private val scaleHit = scales.zipWithIndex.map { case (m, column) =>
    column.U >= activeColumns || m.io.hit
  }.reduce(_ && _)

  private val needsAllocation = !tracker.io.active(incoming.workId)
  private val admit = !rawValid && shapeValid && scaleHit &&
    !pendingRob(incoming.robId) &&
    (!needsAllocation || tracker.io.allocate.ready)

  writes.io.reserveRows.bits := incoming.validRows
  writes.io.reserveRows.valid := io.contextIssue.valid &&
    contexts.io.enq.ready && admit
  contexts.io.enq.valid := io.contextIssue.valid &&
    writes.io.reserveRows.ready && admit
  contexts.io.enq.bits.context := incoming
  contexts.io.enq.bits.scaleSlot := slotWide(slotWidth - 1, 0)
  io.contextIssue.ready := contexts.io.enq.ready &&
    writes.io.reserveRows.ready && admit

  tracker.io.allocate.valid := io.contextIssue.fire && needsAllocation
  tracker.io.allocate.bits := incoming.workId
  tracker.io.issue.valid := io.contextIssue.fire
  tracker.io.issue.bits := incoming.workId
  tracker.io.seal.valid := io.contextIssue.fire && incoming.lastContribution
  tracker.io.seal.bits := incoming.workId
  io.workDone <> tracker.io.done

  // Non-stallable raw output; PRELOAD reserved storage in advance.
  io.rawWrite.foreach(_.ready := true.B)
  private val rowsReceived = RegInit(0.U(log2Ceil(dim + 1).W))
  private val lastRow = rowsReceived + 1.U === head.validRows
  writes.io.enq.valid := rawValid && contexts.io.deq.valid && scaleHit
  writes.io.enq.bits := 0.U.asTypeOf(writes.io.enq.bits)
  writes.io.enq.bits.accBank := rawBank
  writes.io.enq.bits.accRow := raw.addr
  writes.io.enq.bits.accumulate := raw.acc
  writes.io.enq.bits.workId := head.workId
  writes.io.enq.bits.robId := head.robId
  writes.io.enq.bits.completesCommand := lastRow

  for (column <- 0 until dim) {
    val scu = Module(new PoTalSCU(rawBits))
    scu.io.partial := raw.data(column)(0).asUInt(rawBits - 1, 0).asSInt
    scu.io.carrier := scales(column).io.carrier
    writes.io.enq.bits.data(column) := scu.io.result
    when(rawValid && column.U < head.validColumns) {
      assert(scu.io.carrierValid, "PoTal SCU invalid carrier")
      for (byte <- 1 until 4) {
        assert(raw.mask(4 * column + byte) === raw.mask(4 * column),
          "PoTal raw ACC lane mask must be 4 uniform bytes")
      }
    }
  }
  writes.io.enq.bits.mask := VecInit((0 until dim).map { column =>
    raw.mask(column * 4) && column.U < head.validColumns
  }).asUInt

  contexts.io.deq.ready := writes.io.enq.fire && lastRow
  when(writes.io.enq.fire) {
    rowsReceived := Mux(lastRow, 0.U, rowsReceived + 1.U)
  }
  assert(PopCount(VecInit(io.rawWrite.map(_.valid))) <= 1.U,
    "PoTal: multiple raw output banks in one cycle")
  when(rawValid) {
    assert(contexts.io.deq.valid, "PoTal raw output missing context")
    assert(scaleHit, "PoTal raw output scale generation miss")
    assert(writes.io.enq.ready, "PoTal raw output lacked reserved row")
    assert(raw.acc === !head.firstContribution,
      "PoTal ACC accumulation mode differs from accepted context")
  }

  // Accepted writes traverse the physical ACC pipeline.
  private val commitValid = RegInit(VecInit(
    Seq.fill(config.acc_latency)(false.B)))
  private val commitEntry = Reg(Vec(config.acc_latency,
    chiselTypeOf(writes.io.deq.bits)))
  private val queued = writes.io.deq.bits
  private val addressBusy = (0 until config.acc_latency).map { i =>
    commitValid(i) &&
    commitEntry(i).accBank === queued.accBank &&
    commitEntry(i).accRow === queued.accRow
  }.reduce(_ || _)

  for (bank <- 0 until config.acc_banks) {
    val out = io.scaledWrite(bank)
    out.valid := writes.io.deq.valid &&
      queued.accBank === bank.U && !addressBusy
    out.bits := 0.U.asTypeOf(out.bits)
    out.bits.addr := queued.accRow
    out.bits.acc := queued.accumulate
    for (column <- 0 until dim) {
      out.bits.data(column)(0) := queued.data(column)
      for (byte <- 0 until 4) {
        out.bits.mask(column * 4 + byte) := queued.mask(column)
      }
    }
  }
  writes.io.deq.ready := !addressBusy &&
    io.scaledWrite(queued.accBank).ready

  commitValid(0) := writes.io.deq.fire
  commitEntry(0) := queued
  for (i <- 1 until config.acc_latency) {
    commitValid(i) := commitValid(i - 1)
    commitEntry(i) := commitEntry(i - 1)
  }
  private val completedRow = commitEntry.last
  private val completedCommand =
    commitValid.last && completedRow.completesCommand
  tracker.io.commit.valid := completedCommand
  tracker.io.commit.bits := completedRow.workId

  // Non-scaled EX commands still produce immediate ROB completions.
  private val immediate = Module(new Queue(UInt(robWidth.W), robEntries))
  private val delayed = Module(new Queue(UInt(robWidth.W), robEntries))
  immediate.io.enq.valid := io.rawCompletion.valid &&
    !pendingRob(io.rawCompletion.bits)
  immediate.io.enq.bits := io.rawCompletion.bits

  private val readyRob = VecInit((0 until robEntries).map { id =>
    pendingRob(id) && rawDone(id) && writeDone(id)
  })
  delayed.io.enq.valid := readyRob.asUInt.orR
  delayed.io.enq.bits := PriorityEncoder(readyRob.asUInt)

  private val completionArb = Module(new RRArbiter(UInt(robWidth.W), 2))
  completionArb.io.in(0) <> immediate.io.deq
  completionArb.io.in(1) <> delayed.io.deq
  io.robCompletion <> completionArb.io.out

  when(io.contextIssue.fire) {
    pendingRob(incoming.robId) := true.B
    rawDone(incoming.robId) := false.B
    writeDone(incoming.robId) := false.B
  }
  when(io.rawCompletion.valid && pendingRob(io.rawCompletion.bits)) {
    rawDone(io.rawCompletion.bits) := true.B
  }
  when(completedCommand) {
    writeDone(completedRow.robId) := true.B
  }
  when(delayed.io.enq.fire) {
    pendingRob(delayed.io.enq.bits) := false.B
  }

  io.error.scaleMiss := io.contextIssue.valid && !rawValid && !scaleHit
  io.error.invalidScale := scales.map(_.io.invalidWrite).reduce(_ || _)
  io.error.scaleConflict := scales.map(_.io.writeConflict).reduce(_ || _)
  io.error.releaseMiss := scales.map(_.io.releaseMiss).reduce(_ || _)
  io.error.rawWithoutContext := rawValid && !contexts.io.deq.valid
  io.error.reservationError := writes.io.missingReservation
  io.error.completionOverflow := immediate.io.enq.valid &&
    !immediate.io.enq.ready
  io.error.workProtocol := tracker.io.duplicateAllocate ||
    tracker.io.invalidIssue || tracker.io.invalidSeal ||
    tracker.io.commitUnderflow || tracker.io.issueOverflow

  assert(!io.error.completionOverflow,
    "PoTal: non-backpressurable raw completion FIFO overflow")
  assert(!io.error.workProtocol, "PoTal: invalid work completion protocol")

  io.pipelineDrained := !contexts.io.deq.valid &&
    writes.io.reserved === 0.U && writes.io.queued === 0.U &&
    !commitValid.asUInt.orR && !pendingRob.asUInt.orR &&
    !immediate.io.deq.valid && !delayed.io.deq.valid
  io.drained := io.pipelineDrained && !tracker.io.active.asUInt.orR
}

