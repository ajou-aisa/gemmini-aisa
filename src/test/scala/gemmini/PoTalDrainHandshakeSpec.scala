
package gemmini

import chisel3._
import chisel3.util._
import chisel3.iotesters._
import Arithmetic.SIntArithmetic

import scala.collection.mutable.ArrayBuffer

// W3-2c-9d-2: Test-only access to the physical ACC SRAM write stage.
// Matches the non-singleported AccumulatorMem configuration used by
// PoTalWritebackRMWSpec; production RTL remains unchanged.
final class PoTalDrainObservedAccumulatorMem(
  entries: Int,
  rowType: Vec[Vec[SInt]],
  latency: Int
) extends AccumulatorMem[SInt, Float](
  entries, rowType,
  (x: SInt, _: Float) => x,
  Float(8, 24),
  false, 2, false,
  latency, SInt(32.W), false
) {
  val commitPulse = IO(Output(Bool()))
  commitPulse := pipelined_writes(latency - 1).valid
}

// Real PoTalScaleCommandIngress -> real PoTalWriteback ->
// real ACC SRAM banks. Only the older-Gemmini drain predicate and
// the physical ACC backpressure are test-controlled.
final class PoTalDrainHandshakeHarness extends Module {
  val config = GemminiConfigs.defaultConfig.copy(
    dataflow = Dataflow.WS,
    acc_capacity = CapacityInMatrices(2),
    acc_latency = 2,
    acc_singleported = false,
    reservation_station_entries_ex = 2
  )

  private val dim = config.DIM
  require(dim == 16, "W3-2c-9d-2 test assumes DIM16")
  private val rowType = Vec(dim, Vec(1, SInt(32.W)))
  private def writeType =
    new AccumulatorWriteReq(config.acc_bank_entries, rowType)

  val io = IO(new Bundle {
    // CPU-facing control-command ingress.
    val cmd = Flipped(Decoupled(new PoTalScaleCommand))
    val resp = Decoupled(new PoTalScaleCommandResponse)

    // Test-only overrides of actual Controller-side conditions.
    val olderGemminiDrained = Input(Bool())
    val holdAccWrites = Input(Bool())

    // Real writeback inputs and completion outputs.
    val contextIssue = Flipped(Decoupled(new PoTalOutputContext(
      dim, config.ROB_ID_WIDTH, 2, 8)))
    val rawWrite = Flipped(Vec(config.acc_banks, Decoupled(writeType)))
    val rawCompletion = Flipped(Valid(UInt(config.ROB_ID_WIDTH.W)))
    val robCompletion = Decoupled(UInt(config.ROB_ID_WIDTH.W))
    val workDone = Decoupled(UInt(2.W))

    // Read the actual SRAM, not a shadow model.
    val readReq = Flipped(Vec(config.acc_banks,
      Decoupled(UInt(log2Ceil(config.acc_bank_entries).W))))
    val readResp = Vec(config.acc_banks, Decoupled(rowType.cloneType))

    // Test-only observations.
    val accWriteAccepted = Output(Vec(config.acc_banks, Bool()))
    val accSramCommit = Output(Vec(config.acc_banks, Bool()))
    val scaledPending = Output(Vec(config.acc_banks, Bool()))
    val pipelineDrained = Output(Bool())
    val drained = Output(Bool())
    val allDrained = Output(Bool())
    val scaleLoadFire = Output(Bool())
    val scaleReleaseFire = Output(Bool())
    val fatalError = Output(Bool())
  })

  val ingress = Module(new PoTalScaleCommandIngress(
    dim = dim, scaleEntries = 8, generationWidth = 8))

  val wb = Module(new PoTalWriteback(
    config,
    scaleEntries = 8,
    generationWidth = 8,
    workEntries = 4,
    contextDepth = 2,
    queueDepth = 32
  ))

  io.cmd <> ingress.io.cmd
  io.resp <> ingress.io.resp

  // Same connections as Controller.scala: SCALE commands access the
  // ScaleMemory physically owned by PoTalWriteback.
  wb.io.scaleLoad <> ingress.io.scaleLoad
  wb.io.scaleRelease <> ingress.io.scaleRelease
  ingress.io.scaleConflict := wb.io.error.scaleConflict
  ingress.io.releaseMiss := wb.io.error.releaseMiss

  val allDrained = io.olderGemminiDrained && wb.io.drained
  ingress.io.releaseSafe := allDrained
  ingress.io.fenceDrained := allDrained
  io.allDrained := allDrained
  io.scaleLoadFire := ingress.io.scaleLoad.fire
  io.scaleReleaseFire := ingress.io.scaleRelease.fire

  io.contextIssue <> wb.io.contextIssue
  io.rawWrite <> wb.io.rawWrite
  wb.io.rawCompletion := io.rawCompletion
  io.robCompletion <> wb.io.robCompletion
  io.workDone <> wb.io.workDone
  io.pipelineDrained := wb.io.pipelineDrained
  io.drained := wb.io.drained

  // SCALE_CONFLICT and RELEASE_MISS are legitimate CPU-command status
  // conditions. Exclude only those from the fatal datapath error check.
  io.fatalError :=
    wb.io.error.scaleMiss ||
    wb.io.error.invalidScale ||
    wb.io.error.rawWithoutContext ||
    wb.io.error.reservationError ||
    wb.io.error.completionOverflow ||
    wb.io.error.workProtocol

  val shared = Module(new AccPipeShared(
    latency = config.acc_latency - 1,
    t = rowType,
    banks = config.acc_banks,
    saturateInt32 = true
  ))

  val memories = Seq.fill(config.acc_banks) {
    Module(new PoTalDrainObservedAccumulatorMem(
      config.acc_bank_entries, rowType, config.acc_latency))
  }

  for ((mem, bank) <- memories.zipWithIndex) {
    val scaled = wb.io.scaledWrite(bank)
    mem.io.write.valid := scaled.valid && !io.holdAccWrites
    mem.io.write.bits := scaled.bits
    scaled.ready := mem.io.write.ready && !io.holdAccWrites

    io.scaledPending(bank) := scaled.valid
    io.accWriteAccepted(bank) := mem.io.write.fire
    io.accSramCommit(bank) := mem.commitPulse

    shared.io.in_sel(bank) := mem.io.adder.valid
    shared.io.ina(bank) := mem.io.adder.op1
    shared.io.inb(bank) := mem.io.adder.op2
    mem.io.adder.sum := shared.io.out

    mem.io.read.req.valid := io.readReq(bank).valid
    mem.io.read.req.bits := 0.U.asTypeOf(mem.io.read.req.bits)
    mem.io.read.req.bits.addr := io.readReq(bank).bits
    mem.io.read.req.bits.full := true.B
    mem.io.read.req.bits.fromDMA := true.B
    io.readReq(bank).ready := mem.io.read.req.ready
    io.readResp(bank).valid := mem.io.read.resp.valid
    io.readResp(bank).bits := mem.io.read.resp.bits.data
    mem.io.read.resp.ready := io.readResp(bank).ready
  }

  assert(PopCount(shared.io.in_sel) <= 1.U,
    "PoTal drain test: concurrent RMW violates shared ACC adder")
}

final class PoTalDrainHandshakeTester(
  dut: PoTalDrainHandshakeHarness,
  scenario: Int
) extends PeekPokeTester(dut) {
  private val dim = dut.config.DIM
  private val banks = dut.config.acc_banks
  private val WRITE = 23
  private val RELEASE = 24
  private val FENCE = 25
  private val OK = 0
  private val SCALE_CONFLICT = 3
  private val rd = 11

  private var cycle = 0
  private val accepted = ArrayBuffer.empty[(Int, Int)]
  private val committed = ArrayBuffer.empty[(Int, Int)]
  private val robEvents = ArrayBuffer.empty[(Int, Int)]
  private val workEvents = ArrayBuffer.empty[(Int, Int)]
  private val cmdEvents = ArrayBuffer.empty[(Int, Int)]
  private val respEvents = ArrayBuffer.empty[(Int, Int, Int)]
  private val releaseEvents = ArrayBuffer.empty[Int]

  poke(dut.io.cmd.valid, 0)
  poke(dut.io.resp.ready, 1)
  poke(dut.io.olderGemminiDrained, 1)
  poke(dut.io.holdAccWrites, 0)
  poke(dut.io.contextIssue.valid, 0)
  poke(dut.io.rawCompletion.valid, 0)
  poke(dut.io.rawCompletion.bits, 0)
  poke(dut.io.robCompletion.ready, 1)
  poke(dut.io.workDone.ready, 1)

  for (bank <- 0 until banks) {
    val w = dut.io.rawWrite(bank)
    poke(w.valid, 0)
    poke(w.bits.addr, 0)
    poke(w.bits.acc, 0)
    for (col <- 0 until dim)
      poke(w.bits.data(col)(0), 0)
    for (byte <- 0 until 4 * dim)
      poke(w.bits.mask(byte), 0)

    poke(dut.io.readReq(bank).valid, 0)
    poke(dut.io.readReq(bank).bits, 0)
    poke(dut.io.readResp(bank).ready, 0)
  }
  reset(3)

  // Every observed event is sampled BEFORE the active SRAM clock edge.
  private def tick(count: Int = 1): Unit = {
    for (_ <- 0 until count) {
      expect(dut.io.fatalError, 0)

      val acceptedBanks = (0 until banks).filter { bank =>
        peek(dut.io.accWriteAccepted(bank)) == 1
      }
      assert(acceptedBanks.size <= 1,
        "Multiple physical ACC writes accepted in one cycle")
      acceptedBanks.foreach(bank => accepted += ((cycle, bank)))

      for (bank <- 0 until banks) {
        if (peek(dut.io.accSramCommit(bank)) == 1)
          committed += ((cycle, bank))
      }
      assert(committed.size <= accepted.size,
        "Physical SRAM commit without an accepted write")

      if (peek(dut.io.robCompletion.valid) == 1 &&
          peek(dut.io.robCompletion.ready) == 1) {
        assert(committed.nonEmpty && cycle > committed.last._1,
          "ROB completion did not follow physical SRAM commit")
        robEvents += ((cycle, peek(dut.io.robCompletion.bits).toInt))
      }

      if (peek(dut.io.workDone.valid) == 1 &&
          peek(dut.io.workDone.ready) == 1) {
        assert(committed.nonEmpty && cycle > committed.last._1,
          "Work completion did not follow physical SRAM commit")
        workEvents += ((cycle, peek(dut.io.workDone.bits).toInt))
      }

      if (peek(dut.io.cmd.valid) == 1 &&
          peek(dut.io.cmd.ready) == 1)
        cmdEvents += ((cycle, peek(dut.io.cmd.bits.funct).toInt))

      if (peek(dut.io.scaleReleaseFire) == 1)
        releaseEvents += cycle

      if (peek(dut.io.resp.valid) == 1 &&
          peek(dut.io.resp.ready) == 1)
        respEvents += ((cycle,
          peek(dut.io.resp.bits.rd).toInt,
          peek(dut.io.resp.bits.status).toInt))

      step(1)
      cycle += 1
    }
  }

  private def offer(
    funct: Int,
    slot: Int = 0,
    generation: Int = 0,
    column: Int = 0,
    carrier: BigInt = BigInt(0)
  ): Unit = {
    val rs1 = BigInt(slot) |
      (BigInt(generation) << 8) |
      (BigInt(column) << 16)
    poke(dut.io.cmd.bits.funct, funct)
    poke(dut.io.cmd.bits.rs1, rs1)
    poke(dut.io.cmd.bits.rs2, carrier)
    poke(dut.io.cmd.bits.rd, rd)
    poke(dut.io.cmd.bits.xd, 1)
    poke(dut.io.cmd.valid, 1)
  }

  private def awaitResponse(previousCount: Int, expectedStatus: Int): Unit = {
    var waited = 0
    while (respEvents.size == previousCount && waited < 20) {
      tick()
      waited += 1
    }
    assert(respEvents.size == previousCount + 1,
      s"Missing or duplicate response: $respEvents")
    val (responseCycle, responseRd, status) = respEvents.last
    assert(responseRd == rd && status == expectedStatus,
      s"Unexpected response at cycle $responseCycle: rd=$responseRd status=$status")
    expect(dut.io.resp.valid, 0)
  }

  private def immediateCommand(
    funct: Int,
    slot: Int = 0,
    generation: Int = 0,
    column: Int = 0,
    carrier: BigInt = BigInt(0),
    status: Int = OK
  ): Unit = {
    val previous = respEvents.size
    offer(funct, slot, generation, column, carrier)
    expect(dut.io.cmd.ready, 1)
    tick() // Accept the command.
    poke(dut.io.cmd.valid, 0)
    awaitResponse(previous, status)
  }

  // The caller leaves the offered command valid for the entire stall.
  private def expectBlocked(count: Int): Unit = {
    for (_ <- 0 until count) {
      expect(dut.io.allDrained, 0)
      expect(dut.io.cmd.ready, 0)
      expect(dut.io.resp.valid, 0)
      expect(dut.io.scaleReleaseFire, 0)
      tick()
    }
  }

  private def awaitReady(limit: Int = 240): Unit = {
    var waited = 0
    while (peek(dut.io.cmd.ready) == 0 && waited < limit) {
      tick()
      waited += 1
    }
    assert(waited < limit, "Timed out waiting for drained command admission")
    expect(dut.io.allDrained, 1)
  }

  private def acceptHeldCommand(expectedStatus: Int): Int = {
    val previous = respEvents.size
    expect(dut.io.cmd.valid, 1)
    expect(dut.io.cmd.ready, 1)
    val fireCycle = cycle
    tick() // Control-command handshake.
    poke(dut.io.cmd.valid, 0)
    awaitResponse(previous, expectedStatus)
    fireCycle
  }

  private def initializeScale(): Unit = {
    // One 8-slot entry for two active output columns, generation 1.
    // col0 shift=1 (x2), col1 shift=0 (x1).
    immediateCommand(WRITE, slot = 0, generation = 1, column = 0,
      carrier = BigInt(1))
    immediateCommand(WRITE, slot = 0, generation = 1, column = 1,
      carrier = BigInt(0))
  }

  private def issueSingleRow(): Unit = {
    val c = dut.io.contextIssue
    poke(c.bits.robId, 1)
    poke(c.bits.workId, 0)
    poke(c.bits.blockIndex, 0)
    poke(c.bits.scaleBase, 0)
    poke(c.bits.outputTileCount, 1)
    poke(c.bits.outputTileIndex, 0)
    poke(c.bits.scaleGeneration, 1)
    poke(c.bits.validRows, 1)
    poke(c.bits.validColumns, 2)
    poke(c.bits.firstContribution, 1)
    poke(c.bits.lastContribution, 1)
    poke(c.valid, 1)
    expect(c.ready, 1)
    tick()
    poke(c.valid, 0)
  }

  private def issueRaw(bank: Int): Unit = {
    val w = dut.io.rawWrite(bank)
    poke(w.bits.addr, 3)
    poke(w.bits.acc, 0)
    for (col <- 0 until dim)
      poke(w.bits.data(col)(0),
        if (col == 0) BigInt(7)
        else if (col == 1) BigInt(-3)
        else BigInt(0))
    for (byte <- 0 until 4 * dim)
      poke(w.bits.mask(byte), if (byte < 8) 1 else 0)

    poke(w.valid, 1)
    expect(w.ready, 1)
    tick()
    poke(w.valid, 0)

    poke(dut.io.rawCompletion.bits, 1)
    poke(dut.io.rawCompletion.valid, 1)
    tick()
    poke(dut.io.rawCompletion.valid, 0)
  }

  private def launchStalledWrite(bank: Int): Unit = {
    poke(dut.io.holdAccWrites, 1)
    issueSingleRow()
    issueRaw(bank)

    var waited = 0
    while (peek(dut.io.scaledPending(bank)) == 0 && waited < 20) {
      tick()
      waited += 1
    }
    assert(waited < 20, "Scaled write did not reach ACC backpressure")
    expect(dut.io.scaledPending(bank), 1)
    expect(dut.io.pipelineDrained, 0)
    expect(dut.io.drained, 0)
    tick(5)
    assert(accepted.isEmpty && committed.isEmpty,
      "ACC write bypassed test backpressure")
    assert(robEvents.isEmpty && workEvents.isEmpty,
      "Completion occurred while ACC write was blocked")
  }

  private def verifyWriteCompleted(bank: Int): Unit = {
    expect(dut.io.drained, 1)
    assert(accepted.size == 1 && committed.size == 1,
      s"Expected exactly one ACC accept and commit: $accepted / $committed")
    assert(accepted.head._2 == bank && committed.head._2 == bank)
    assert(committed.head._1 - accepted.head._1 == dut.config.acc_latency,
      s"Incorrect SRAM write latency: $accepted / $committed")

    assert(robEvents.map(_._2).toSeq == Seq(1),
      s"ROB completion missing or duplicated: $robEvents")
    assert(workEvents.map(_._2).toSeq == Seq(0),
      s"Work completion missing or duplicated: $workEvents")
    assert(robEvents.head._1 > committed.head._1)
    assert(workEvents.head._1 > committed.head._1)
  }

  private def readExpect(bank: Int): Unit = {
    poke(dut.io.readReq(bank).bits, 3)
    poke(dut.io.readReq(bank).valid, 1)
    var waited = 0
    while (peek(dut.io.readReq(bank).ready) == 0 && waited < 40) {
      tick()
      waited += 1
    }
    assert(waited < 40, "ACC SRAM read request timeout")
    tick()
    poke(dut.io.readReq(bank).valid, 0)

    waited = 0
    while (peek(dut.io.readResp(bank).valid) == 0 && waited < 40) {
      tick()
      waited += 1
    }
    assert(waited < 40, "ACC SRAM read response timeout")
    // 7 * (1 << 1) == 14; -3 * (1 << 0) == -3.
    expect(dut.io.readResp(bank).bits(0)(0), 14)
    expect(dut.io.readResp(bank).bits(1)(0), -3)
    poke(dut.io.readResp(bank).ready, 1)
    tick()
    poke(dut.io.readResp(bank).ready, 0)
  }

  private def checkNoDuplicateCompletions(): Unit = {
    tick(5)
    assert(accepted.size == 1 && committed.size == 1)
    assert(robEvents.size == 1 && workEvents.size == 1,
      s"Duplicate completion: $robEvents / $workEvents")
  }

  initializeScale()

  if (scenario == 0) {
    // RELEASE waits for real writeback + physical SRAM commit.
    launchStalledWrite(bank = 0)
    offer(RELEASE, slot = 0, generation = 1, column = 0)
    expectBlocked(6)
    expect(dut.io.drained, 0)
    poke(dut.io.holdAccWrites, 0)
    awaitReady()
    verifyWriteCompleted(bank = 0)
    val releaseCycle = acceptHeldCommand(OK)
    assert(releaseEvents.toSeq == Seq(releaseCycle))
    assert(releaseCycle > committed.head._1,
      "RELEASE accepted before physical SRAM commit")

    // Reallocation is possible only after RELEASE; the generation changes.
    immediateCommand(WRITE, slot = 0, generation = 2, column = 0,
      carrier = BigInt(2))
    readExpect(bank = 0)
    checkNoDuplicateCompletions()
    println(s"W3-2c-9d-2 scenario 0 PASS: RELEASE cycle=$releaseCycle " +
      s"SRAM commit=${committed.head._1}")

  } else if (scenario == 1) {
    // FENCE must wait for the same real writeback drain, but must NOT
    // release or mutate the stored ScaleMemory ownership.
    launchStalledWrite(bank = 1)
    offer(FENCE)
    expectBlocked(6)
    poke(dut.io.holdAccWrites, 0)
    awaitReady()
    verifyWriteCompleted(bank = 1)
    val fenceCycle = acceptHeldCommand(OK)
    assert(fenceCycle > committed.head._1)
    assert(releaseEvents.isEmpty, "FENCE unexpectedly released a scale")

    // An attempted reallocation MUST still conflict after FENCE.
    immediateCommand(WRITE, slot = 0, generation = 2, column = 0,
      carrier = BigInt(2), status = SCALE_CONFLICT)
    immediateCommand(RELEASE, slot = 0, generation = 1, column = 0)
    immediateCommand(WRITE, slot = 0, generation = 2, column = 0,
      carrier = BigInt(2))
    readExpect(bank = 1)
    checkNoDuplicateCompletions()
    println(s"W3-2c-9d-2 scenario 1 PASS: FENCE cycle=$fenceCycle " +
      s"SRAM commit=${committed.head._1}")

  } else if (scenario == 2) {
    // Even after WB drain, the older Gemmini predicate alone must stall
    // both RELEASE and FENCE. Exercise two independent drain blockers.
    poke(dut.io.olderGemminiDrained, 0)
    launchStalledWrite(bank = 0)
    offer(RELEASE, slot = 0, generation = 1, column = 0)
    expectBlocked(4) // olderGemminiDrained=0, wb.drained=0
    poke(dut.io.holdAccWrites, 0)

    var waited = 0
    while (peek(dut.io.drained) == 0 && waited < 240) {
      expect(dut.io.cmd.ready, 0)
      tick()
      waited += 1
    }
    assert(waited < 240, "Writeback did not drain")
    verifyWriteCompleted(bank = 0)
    expectBlocked(6) // olderGemminiDrained=0, wb.drained=1

    poke(dut.io.olderGemminiDrained, 1)
    expect(dut.io.allDrained, 1)
    val releaseCycle = acceptHeldCommand(OK)
    assert(releaseCycle > committed.head._1)

    poke(dut.io.olderGemminiDrained, 0)
    offer(FENCE)
    expect(dut.io.drained, 1)
    expectBlocked(6) // FENCE blocked solely by olderGemminiDrained
    poke(dut.io.olderGemminiDrained, 1)
    val fenceCycle = acceptHeldCommand(OK)
    assert(fenceCycle > releaseCycle)
    assert(releaseEvents.toSeq == Seq(releaseCycle))
    readExpect(bank = 0)
    checkNoDuplicateCompletions()
    println(s"W3-2c-9d-2 scenario 2 PASS: " +
      s"RELEASE cycle=$releaseCycle FENCE cycle=$fenceCycle")

  } else {
    throw new IllegalArgumentException(s"Unknown drain scenario $scenario")
  }
}

class PoTalDrainHandshakeSpec extends ChiselFlatSpec {
  behavior of "PoTal RELEASE/FENCE under real writeback drain"

  for (scenario <- 0 until 3) {
    it should s"preserve drain sequencing in scenario $scenario" in {
      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir", s"test_run_dir/potal-drain-handshake-$scenario"
        ),
        () => new PoTalDrainHandshakeHarness
      ) { dut =>
        new PoTalDrainHandshakeTester(dut, scenario)
      } should be (true)
    }
  }
}

