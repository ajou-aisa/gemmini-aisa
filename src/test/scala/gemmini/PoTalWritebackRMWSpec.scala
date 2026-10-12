package gemmini

import chisel3._
import chisel3.util._
import chisel3.iotesters._
import Arithmetic.SIntArithmetic

import scala.collection.mutable.ArrayBuffer

// W2: PoTalWriteback -> two real AccumulatorMem banks -> AccPipeShared(Sat32).
// This intentionally does NOT instantiate the full Scratchpad/TileLink subsystem.
final class PoTalWritebackRMWHarness extends Module {
  val config = GemminiConfigs.defaultConfig.copy(
    dataflow = Dataflow.WS,
    acc_capacity = CapacityInMatrices(2),
    acc_latency = 2,
    acc_singleported = false,
    reservation_station_entries_ex = 2
  )

  private val dim = config.DIM
  private val rowType = Vec(dim, Vec(1, SInt(32.W)))
  private def writeType =
    new AccumulatorWriteReq(config.acc_bank_entries, rowType)

  val io = IO(new Bundle {
    val contextIssue = Flipped(Decoupled(new PoTalOutputContext(
      dim, config.ROB_ID_WIDTH, 2, 4)))
    val rawWrite = Flipped(Vec(config.acc_banks, Decoupled(writeType)))
    val rawCompletion = Flipped(Valid(UInt(config.ROB_ID_WIDTH.W)))
    val scaleLoad = Flipped(Decoupled(new PoTalScaleLoad(dim, 4, 4)))
    val scaleRelease = Flipped(Decoupled(new PoTalScaleRelease(dim, 4, 4)))
    val robCompletion = Decoupled(UInt(config.ROB_ID_WIDTH.W))
    val workDone = Decoupled(UInt(2.W))

    val readReq = Flipped(Vec(config.acc_banks,
      Decoupled(UInt(log2Ceil(config.acc_bank_entries).W))))
    val readResp = Vec(config.acc_banks, Decoupled(rowType.cloneType))

    val accWriteAccepted = Output(Vec(config.acc_banks, Bool()))
    val pipelineDrained = Output(Bool())
    val drained = Output(Bool())
    val error = Output(Bool())
  })

  val writeback = Module(new PoTalWriteback(
    config,
    scaleEntries = 4,
    generationWidth = 4,
    workEntries = 4,
    contextDepth = 2,
    queueDepth = 32
  ))

  io.contextIssue <> writeback.io.contextIssue
  io.rawWrite <> writeback.io.rawWrite
  writeback.io.rawCompletion := io.rawCompletion
  io.scaleLoad <> writeback.io.scaleLoad
  io.scaleRelease <> writeback.io.scaleRelease
  io.robCompletion <> writeback.io.robCompletion
  io.workDone <> writeback.io.workDone
  io.pipelineDrained := writeback.io.pipelineDrained
  io.drained := writeback.io.drained
  io.error := writeback.io.error.asUInt.orR

  val shared = Module(new AccPipeShared(
    latency = config.acc_latency - 1,
    t = rowType,
    banks = config.acc_banks,
    saturateInt32 = true
  ))

  val memories = Seq.fill(config.acc_banks) {
    Module(new AccumulatorMem(
      config.acc_bank_entries,
      rowType,
      (x: SInt, _: Float) => x,
      Float(8, 24),
      false, 2, false,
      config.acc_latency,
      SInt(32.W),
      false
    ))
  }

  for ((mem, bank) <- memories.zipWithIndex) {
    // The writeback receives REAL ACC backpressure and cannot bypass SRAM.
    mem.io.write <> writeback.io.scaledWrite(bank)
    io.accWriteAccepted(bank) := mem.io.write.fire

    // Exactly the RMW adder-sharing topology in Scratchpad.scala.
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
    "PoTal W2: concurrent ACC bank RMW violates shared adder")
}

class PoTalWritebackRMWTester(
  dut: PoTalWritebackRMWHarness,
  scenario: Int
) extends PeekPokeTester(dut) {
  private val dim = 16
  private val zeroCarrier = BigInt("80000000", 16)
  private val max32 = (BigInt(1) << 31) - 1

  private val completedRob = ArrayBuffer.empty[Int]
  private val completedWork = ArrayBuffer.empty[Int]
  private val acceptedWrites = Array.fill(2)(0)

  poke(dut.io.contextIssue.valid, 0)
  poke(dut.io.rawCompletion.valid, 0)
  poke(dut.io.rawCompletion.bits, 0)
  poke(dut.io.scaleLoad.valid, 0)
  poke(dut.io.scaleRelease.valid, 0)
  poke(dut.io.robCompletion.ready, 1)
  poke(dut.io.workDone.ready, 1)

  for (bank <- 0 until 2) {
    poke(dut.io.rawWrite(bank).valid, 0)
    poke(dut.io.readReq(bank).valid, 0)
    poke(dut.io.readReq(bank).bits, 0)
    poke(dut.io.readResp(bank).ready, 0)
    poke(dut.io.rawWrite(bank).bits.addr, 0)
    poke(dut.io.rawWrite(bank).bits.acc, 0)
    for (col <- 0 until dim)
      poke(dut.io.rawWrite(bank).bits.data(col)(0), 0)
    for (byte <- 0 until dim * 4)
      poke(dut.io.rawWrite(bank).bits.mask(byte), 0)
  }
  reset(2)

  // Observe handshakes BEFORE the rising edge; this catches completions
  // even when they occur during the load/issue/raw helper procedures.
  private def tick(count: Int = 1): Unit = {
    for (_ <- 0 until count) {
      expect(dut.io.error, 0)
      for (bank <- 0 until 2)
        if (peek(dut.io.accWriteAccepted(bank)) == 1)
          acceptedWrites(bank) += 1
      if (peek(dut.io.robCompletion.valid) == 1 &&
          peek(dut.io.robCompletion.ready) == 1) {
        val id = peek(dut.io.robCompletion.bits).toInt
        assert(acceptedWrites(scenario) >= id,
          s"ROB $id completed before its ACC write was accepted")
        completedRob += id
      }
      if (peek(dut.io.workDone.valid) == 1 &&
          peek(dut.io.workDone.ready) == 1) {
        assert(acceptedWrites(scenario) == 3,
          "workDone occurred before all ACC writes were accepted")
        completedWork += peek(dut.io.workDone.bits).toInt
      }
      step(1)
    }
  }

  private def load(col: Int, block: Int, carrier: BigInt): Unit = {
    poke(dut.io.scaleLoad.bits.column, col)
    poke(dut.io.scaleLoad.bits.address, block)
    poke(dut.io.scaleLoad.bits.generation, 1)
    poke(dut.io.scaleLoad.bits.carrier, carrier)
    poke(dut.io.scaleLoad.valid, 1)
    expect(dut.io.scaleLoad.ready, 1)
    tick()
    poke(dut.io.scaleLoad.valid, 0)
  }

  private def issue(
    block: Int,
    rob: Int,
    first: Boolean,
    last: Boolean
  ): Unit = {
    val c = dut.io.contextIssue
    poke(c.bits.robId, rob)
    poke(c.bits.workId, 0)
    poke(c.bits.blockIndex, block)
    poke(c.bits.scaleBase, 0)
    poke(c.bits.outputTileCount, 1)
    poke(c.bits.outputTileIndex, 0)
    poke(c.bits.scaleGeneration, 1)
    poke(c.bits.validRows, 1)
    poke(c.bits.validColumns, 2)
    poke(c.bits.firstContribution, if (first) 1 else 0)
    poke(c.bits.lastContribution, if (last) 1 else 0)
    poke(c.valid, 1)
    expect(c.ready, 1)
    tick()
    poke(c.valid, 0)
  }

  private def raw(
    bank: Int,
    address: Int,
    a: BigInt,
    b: BigInt,
    accumulate: Boolean,
    rob: Int
  ): Unit = {
    val w = dut.io.rawWrite(bank)
    poke(w.bits.addr, address)
    poke(w.bits.acc, if (accumulate) 1 else 0)
    for (col <- 0 until dim) {
      val value = if (col == 0) a else if (col == 1) b else BigInt(0)
      poke(w.bits.data(col)(0), value)
    }
    for (byte <- 0 until 4 * dim)
      poke(w.bits.mask(byte), if (byte < 8) 1 else 0)

    poke(w.valid, 1)
    expect(w.ready, 1)
    tick()
    poke(w.valid, 0)

    poke(dut.io.rawCompletion.bits, rob)
    poke(dut.io.rawCompletion.valid, 1)
    tick()
    poke(dut.io.rawCompletion.valid, 0)
  }

  private def awaitDone(expectBank: Int): Unit = {
    var cycles = 0
    while ((completedRob.size < 3 || completedWork.size < 1 ||
            peek(dut.io.drained) == 0) && cycles < 120) {
      tick()
      cycles += 1
    }
    assert(cycles < 120, "W2 timeout waiting for writeback completion")
    assert(completedRob.sorted.toSeq == Seq(1, 2, 3),
      s"unexpected ROB completions: $completedRob")
    assert(completedWork.toSeq == Seq(0),
      s"unexpected work completion: $completedWork")
    assert(acceptedWrites(expectBank) == 3,
      s"expected three physical ACC write requests on bank $expectBank")
    assert(acceptedWrites(1 - expectBank) == 0,
      "unexpected ACC write on another bank")
    tick(4) // Observe SRAM write settling separately from WB bookkeeping.
  }

  private def readExpect(
    bank: Int,
    address: Int,
    expected0: BigInt,
    expected1: BigInt
  ): Unit = {
    poke(dut.io.readReq(bank).bits, address)
    poke(dut.io.readReq(bank).valid, 1)
    var cycles = 0
    while (peek(dut.io.readReq(bank).ready) == 0 && cycles < 40) {
      tick()
      cycles += 1
    }
    assert(peek(dut.io.readReq(bank).ready) == 1,
      "W2 timeout waiting for ACC read request")
    tick()
    poke(dut.io.readReq(bank).valid, 0)

    cycles = 0
    while (peek(dut.io.readResp(bank).valid) == 0 && cycles < 40) {
      tick()
      cycles += 1
    }
    assert(peek(dut.io.readResp(bank).valid) == 1,
      "W2 timeout waiting for ACC read response")

    // Only lanes 0 and 1 were written; other lanes are uninitialized.
    expect(dut.io.readResp(bank).bits(0)(0), expected0)
    expect(dut.io.readResp(bank).bits(1)(0), expected1)
    poke(dut.io.readResp(bank).ready, 1)
    tick()
    poke(dut.io.readResp(bank).ready, 0)
  }

  if (scenario == 0) {
    // K32 blocks 0,1,2: 1<<31 saturates, then +1, then -1.
    // Stepwise Sat32 must produce INT32_MAX - 1, not INT32_MAX.
    load(0, 0, 31)
    load(1, 0, zeroCarrier)
    load(0, 1, 0)
    load(1, 1, zeroCarrier)
    load(0, 2, 0)
    load(1, 2, zeroCarrier)

    issue(block = 0, rob = 1, first = true, last = false)
    raw(bank = 0, address = 3, a = 1, b = -7,
      accumulate = false, rob = 1)
    issue(block = 1, rob = 2, first = false, last = false)
    raw(bank = 0, address = 3, a = 1, b = 5,
      accumulate = true, rob = 2)
    issue(block = 2, rob = 3, first = false, last = true)
    raw(bank = 0, address = 3, a = -1, b = 10,
      accumulate = true, rob = 3)

    awaitDone(expectBank = 0)
    readExpect(bank = 0, address = 3, expected0 = max32 - 1,
      expected1 = 0)
  } else if (scenario == 1) {
    // Two array passes share K32 block 0's column-wise scale.
    // A later K32 block changes the carrier. Exercise ACC bank 1.
    load(0, 0, 2) // x4
    load(1, 0, 1) // x2
    load(0, 1, 1) // x2
    load(1, 1, zeroCarrier)

    issue(block = 0, rob = 1, first = true, last = false)
    raw(bank = 1, address = 4, a = 3, b = -5,
      accumulate = false, rob = 1)
    issue(block = 0, rob = 2, first = false, last = false)
    raw(bank = 1, address = 4, a = 4, b = 9,
      accumulate = true, rob = 2)
    issue(block = 1, rob = 3, first = false, last = true)
    raw(bank = 1, address = 4, a = -7, b = -100,
      accumulate = true, rob = 3)

    // Column 0: (3*4) + (4*4) + (-7*2) = 14.
    // Column 1: (-5*2) + (9*2) + 0 = 8.
    awaitDone(expectBank = 1)
    readExpect(bank = 1, address = 4, expected0 = 14,
      expected1 = 8)
  } else {
    throw new IllegalArgumentException("unknown W2 scenario")
  }
}

class PoTalWritebackRMWSpec extends ChiselFlatSpec {
  behavior of "PoTal Writeback + physical ACC SRAM"

  for (scenario <- 0 until 2) {
    it should s"preserve scale and Sat32 RMW in scenario $scenario" in {
      Driver.execute(
        Array("--backend-name", "treadle", "--target-dir",
          s"test_run_dir/potal-writeback-rmw-$scenario"),
        () => new PoTalWritebackRMWHarness
      ) { dut =>
        new PoTalWritebackRMWTester(dut, scenario)
      } should be (true)
    }
  }
}

