package gemmini

import chisel3._
import chisel3.util._
import chisel3.iotesters._
import Arithmetic.SIntArithmetic

import scala.collection.mutable.ArrayBuffer


final class PoTalObservedAccumulatorMem(
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
  // Test-only observation of the physical SRAM write stage.
  val commitPulse = IO(Output(Bool()))

  commitPulse :=
    pipelined_writes(latency - 1).valid
}


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
    val accSramCommit = Output(Vec(config.acc_banks, Bool()))
    val pipelineDrained = Output(Bool())
    val drained = Output(Bool())
    val error = Output(Bool())

    // Test-only ACC backpressure control.
    val holdAccWrites = Input(Bool())

    // Scaled writes waiting at the physical ACC interface.
    val scaledPending = Output(
      Vec(config.acc_banks, Bool())
    )
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
    Module(new PoTalObservedAccumulatorMem(
      config.acc_bank_entries,
      rowType,
      config.acc_latency
    ))
  }

  for ((mem, bank) <- memories.zipWithIndex) {
    // The writeback receives REAL ACC backpressure and cannot bypass SRAM.
    val scaled = writeback.io.scaledWrite(bank)
    
    // Force physical ACC backpressure without changing
    // the production PoTalWriteback or AccumulatorMem.
    mem.io.write.valid := scaled.valid && !io.holdAccWrites

    mem.io.write.bits := scaled.bits

    scaled.ready :=mem.io.write.ready && !io.holdAccWrites

    io.scaledPending(bank) := scaled.valid
    io.accWriteAccepted(bank) := mem.io.write.fire 
    io.accSramCommit(bank) := mem.commitPulse

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
  private val acceptedEvents = ArrayBuffer.empty[(Int, Int)]

  private val committedWrites = Array.fill(2)(0)
  private val committedEvents = ArrayBuffer.empty[(Int, Int)]

  private val robCycles = ArrayBuffer.empty[(Int, Int)]

  private val workCycles = ArrayBuffer.empty[(Int, Int)]

  private var testCycle = 0

  poke(dut.io.contextIssue.valid, 0)
  poke(dut.io.rawCompletion.valid, 0)
  poke(dut.io.rawCompletion.bits, 0)
  poke(dut.io.scaleLoad.valid, 0)
  poke(dut.io.scaleRelease.valid, 0)
  poke(dut.io.robCompletion.ready, 1)
  poke(dut.io.workDone.ready, 1)
  poke(dut.io.holdAccWrites, 0)

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

      // Record actual physical ACC write acceptance.
      val firedBanks = (0 until 2).filter { bank =>
        peek(dut.io.accWriteAccepted(bank)) == 1
      }

      // PoTal WB emits at most one scaled row per cycle.
      assert(
        firedBanks.size <= 1,
        "Multiple physical ACC writes in one cycle"
      )

      firedBanks.foreach { bank =>
        acceptedWrites(bank) += 1
        acceptedEvents += ((testCycle, bank))
      }


      // Physical SRAM write stage.
      // Sampled before the rising edge that writes SRAM.
      for (bank <- 0 until 2) {
        if (peek(dut.io.accSramCommit(bank)) == 1) {
          committedWrites(bank) += 1
          committedEvents += ((testCycle, bank))
        }

        assert(
          committedWrites(bank) <= acceptedWrites(bank),
          s"SRAM committed before ACC acceptance on bank $bank"
        )
      }


      if (peek(dut.io.robCompletion.valid) == 1 &&
          peek(dut.io.robCompletion.ready) == 1) {

        val id =
          peek(dut.io.robCompletion.bits).toInt

        if (scenario == 2) {
          assert(id >= 1 && id <= 6)

          val bank = if (id % 2 == 1) 0 else 1
          val required = (id + 1) / 2

          assert(
            committedWrites(bank) >= required,
            s"ROB $id completed before SRAM commit"
          )
        } else {
          assert(
            acceptedWrites(scenario) >= id,
            s"ROB $id completed before ACC write"
          )
        }

        assert(
          !completedRob.contains(id),
          s"Duplicate ROB completion $id"
        )

        completedRob += id
        robCycles += ((id, testCycle))
      }

      if (peek(dut.io.workDone.valid) == 1 &&
          peek(dut.io.workDone.ready) == 1) {

        val work =
          peek(dut.io.workDone.bits).toInt

        val bank =
          if (scenario == 2) work else scenario

        assert(bank >= 0 && bank < 2)

        assert(
          committedWrites(bank) == 3,
          s"Work $work completed before all ACC writes"
        )

        assert(
          !completedWork.contains(work),
          s"Duplicate work completion $work"
        )

        completedWork += work
        workCycles += ((work, testCycle))
      }

      step(1)
      testCycle += 1
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
    last: Boolean,
    work: Int = 0
  ): Unit = {
    val c = dut.io.contextIssue
    poke(c.bits.robId, rob)
    poke(c.bits.workId, work)
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
  } else if (scenario == 2) {

    // ========================================================
    // W3-2c-9b-2: Physical ACC collision stress
    //
    // Work 0 -> Bank 0, Row 3
    // Work 1 -> Bank 1, Row 3
    //
    // Six scaled writes are buffered while physical ACC
    // acceptance is blocked, then released together.
    // ========================================================

    // Block 0: col0 x1, col1 x2
    load(0, 0, 0)
    load(1, 0, 1)

    // Block 1: col0 x2, col1 x1
    load(0, 1, 1)
    load(1, 1, 0)

    // Block 2: col0 x1, col1 x0
    load(0, 2, 0)
    load(1, 2, zeroCarrier)

    // Hold physical ACC writes.
    poke(dut.io.holdAccWrites, 1)

    // --------------------------------------------------------
    // First contributions: ACC overwrite
    // --------------------------------------------------------

    issue(block=0, rob=1, first=true, last=false, work=0)
    raw(bank=0, address=3, a=7, b = -3, accumulate=false, rob=1)

    issue(block=0, rob=2, first=true, last=false, work=1)
    raw(bank=1, address=3, a = -4, b=6, accumulate=false, rob=2)

    // --------------------------------------------------------
    // Second contributions: ACC accumulation
    // --------------------------------------------------------

    issue(block=1, rob=3, first=false, last=false, work=0)
    raw(bank=0, address=3, a=8, b=4, accumulate=true, rob=3)

    issue(block=1, rob=4, first=false, last=false, work=1)
    raw(bank=1, address=3, a=5, b = -7, accumulate=true, rob=4)

    // --------------------------------------------------------
    // Final contributions
    // --------------------------------------------------------

    issue(block=2, rob=5, first=false, last=true, work=0)
    raw(bank=0, address=3, a = -5, b=100, accumulate=true, rob=5)

    issue(block=2, rob=6, first=false, last=true, work=1)
    raw(bank=1, address=3, a=2, b=10, accumulate=true, rob=6)

    // --------------------------------------------------------
    // Verify that ACC backpressure prevents completion.
    // --------------------------------------------------------

    expect(dut.io.scaledPending(0), 1)
    expect(dut.io.scaledPending(1), 0)
    expect(dut.io.pipelineDrained, 0)

    tick(5)

    assert(acceptedWrites.toSeq == Seq(0, 0))
    assert(completedRob.isEmpty)
    assert(completedWork.isEmpty)

    // --------------------------------------------------------
    // Release ACC backpressure.
    // --------------------------------------------------------

    poke(dut.io.holdAccWrites, 0)

    var cycles = 0

    while (
      (completedRob.size < 6 ||
       completedWork.size < 2 ||
       peek(dut.io.drained) == 0) &&
      cycles < 180
    ) {
      tick()
      cycles += 1
    }

    assert(cycles < 180, "Timeout draining ACC collision stress")

    // Exactly three writes accepted by each bank.
    assert(acceptedWrites.toSeq == Seq(3, 3))
    assert(committedWrites.toSeq == Seq(3, 3))

    // FIFO order must be preserved across bank switches.
    assert(acceptedEvents.map(_._2).toSeq == Seq(0, 1, 0, 1, 0, 1), s"Unexpected ACC bank write order: $acceptedEvents")

    // Same-bank, same-row writes must be sufficiently
    // separated for the two-cycle ACC pipeline.
    for (bank <- 0 until 2) {
      val times = acceptedEvents.collect {
        case (cycle, b) if b == bank => cycle
      }

      assert(times.size == 3)

      assert(
        times(1) - times(0) >= 3 &&
        times(2) - times(1) >= 3,
        s"Unsafe same-row reuse on bank $bank: $times"
      )
    }


    // Six physical SRAM commits, preserving FIFO order.
    assert(committedEvents.size == 6)

    assert(
      committedEvents.map(_._2).toSeq ==
        Seq(0, 1, 0, 1, 0, 1),
      s"SRAM commit order mismatch: $committedEvents"
    )

    // Each accepted write enters SRAM after acc_latency cycles.
    for (index <- acceptedEvents.indices) {
      val (acceptCycle, acceptBank) =
        acceptedEvents(index)

      val (commitCycle, commitBank) =
        committedEvents(index)

      assert(acceptBank == commitBank)

      assert(
        commitCycle - acceptCycle ==
          dut.config.acc_latency,
        s"ACC write $index latency mismatch: " +
        s"accepted=$acceptCycle committed=$commitCycle"
      )
    }

    // ROB completion must follow the corresponding SRAM write.
    for ((rob, completeCycle) <- robCycles) {
      val commitCycle = committedEvents(rob - 1)._1

      assert(
        completeCycle > commitCycle,
        s"ROB $rob retired before SRAM commit"
      )
    }

    // Work completion must follow its final SRAM write.
    for ((work, doneCycle) <- workCycles) {
      val finalWriteIndex =
        if (work == 0) 4 else 5

      val commitCycle =
        committedEvents(finalWriteIndex)._1

      assert(
        doneCycle > commitCycle,
        s"Work $work completed before final SRAM write"
      )
    }

    assert(completedRob.sorted.toSeq == (1 to 6).toSeq, s"Unexpected ROB completions: $completedRob")

    assert(completedWork.sorted.toSeq == Seq(0, 1), s"Unexpected work completions: $completedWork")

    expect(dut.io.pipelineDrained, 1)
    expect(dut.io.drained, 1)

    // --------------------------------------------------------
    // Verify actual physical ACC SRAM contents.
    //
    // Bank 0:
    //   col0 = 7 + 8*2 - 5 = 18
    //   col1 = -3*2 + 4 + 0 = -2
    //
    // Bank 1:
    //   col0 = -4 + 5*2 + 2 = 8
    //   col1 = 6*2 - 7 + 0 = 5
    // --------------------------------------------------------

    tick(4)

    readExpect(bank=0, address=3, expected0=18, expected1 = -2)

    readExpect(bank=1, address=3, expected0=8, expected1=5)

    println(s"W3-2c-9b PASS: writes=$acceptedWrites " + s"events=$acceptedEvents")
    } else {
      throw new IllegalArgumentException("unknown W2 scenario")
  }
}

class PoTalWritebackRMWSpec extends ChiselFlatSpec {
  behavior of "PoTal Writeback + physical ACC SRAM"

  for (scenario <- 0 until 3) {
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

