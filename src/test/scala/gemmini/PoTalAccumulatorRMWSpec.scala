
package gemmini

import chisel3._
import chisel3.util._
import chisel3.iotesters._
import Arithmetic.SIntArithmetic

// ============================================================
// Harness: 2 real AccumulatorMem banks + shared AccPipe
// ============================================================

class PoTalAccumulatorRMWHarness(
  saturate: Boolean,
  accLatency: Int = 2
) extends Module {

  require(accLatency >= 2)

  private val bankCount = 2
  private val entries = 16
  private val rowType = Vec(2, Vec(2, SInt(32.W)))

  val io = IO(new Bundle {
    val write = Flipped(Vec(
      bankCount,
      Decoupled(new AccumulatorWriteReq(entries, rowType))
    ))

    val readReq = Flipped(Vec(
      bankCount,
      Decoupled(UInt(log2Ceil(entries).W))
    ))

    val readResp = Vec(
      bankCount,
      Decoupled(rowType.cloneType)
    )

    val adderSelected = Output(Vec(bankCount, Bool()))
  })

  // Identical sharing structure to Scratchpad.scala.
  val sharedAdder = Module(new AccPipeShared(
    latency = accLatency - 1,
    t = rowType,
    banks = bankCount,
    saturateInt32 = saturate
  ))

  val accBanks = Seq.fill(bankCount) {
    Module(new AccumulatorMem(
      n = entries,
      t = rowType,
      scale_func = (x: SInt, _: gemmini.Float) => x,
      scale_t = gemmini.Float(8, 24),
      acc_singleported = false,
      acc_sub_banks = 2,
      use_shared_ext_mem = false,
      acc_latency = accLatency,
      acc_type = SInt(32.W),
      is_dummy = false
    ))
  }

  for ((mem, bank) <- accBanks.zipWithIndex) {

    // ACC write request.
    mem.io.write.valid := io.write(bank).valid
    mem.io.write.bits := io.write(bank).bits
    io.write(bank).ready := mem.io.write.ready

    // Full-width ACC read request.
    mem.io.read.req.valid := io.readReq(bank).valid
    mem.io.read.req.bits :=
      0.U.asTypeOf(mem.io.read.req.bits)

    mem.io.read.req.bits.addr := io.readReq(bank).bits
    mem.io.read.req.bits.full := true.B
    mem.io.read.req.bits.fromDMA := true.B

    io.readReq(bank).ready := mem.io.read.req.ready

    // Read response.
    io.readResp(bank).valid := mem.io.read.resp.valid
    io.readResp(bank).bits := mem.io.read.resp.bits.data
    mem.io.read.resp.ready := io.readResp(bank).ready

    // Shared RMW adder.
    sharedAdder.io.in_sel(bank) := mem.io.adder.valid
    sharedAdder.io.ina(bank) := mem.io.adder.op1
    sharedAdder.io.inb(bank) := mem.io.adder.op2

    mem.io.adder.sum := sharedAdder.io.out

    io.adderSelected(bank) := mem.io.adder.valid
  }

  // The current shared adder supports one active bank
  // per cycle. This assertion detects invalid overlap.
  assert(
    PopCount(sharedAdder.io.in_sel) <= 1.U,
    "PoTal RMW: simultaneous accumulator adder users"
  )
}


// ============================================================
// Independent numerical reference
// ============================================================

object PoTalRMWGolden {

  val max32: BigInt = (BigInt(1) << 31) - 1
  val min32: BigInt = -(BigInt(1) << 31)
  val mod32: BigInt = BigInt(1) << 32
  val mask32: BigInt = mod32 - 1

  def add(
    a: BigInt,
    b: BigInt,
    saturate: Boolean
  ): BigInt = {

    val sum = a + b

    if (saturate) {
      sum.max(min32).min(max32)
    } else {
      val bits = sum & mask32
      if (bits.testBit(31)) bits - mod32
      else bits
    }
  }

  def update(
    previous: Seq[BigInt],
    input: Seq[BigInt],
    accumulate: Boolean,
    lanes: Set[Int],
    saturate: Boolean
  ): Seq[BigInt] = {

    previous.indices.map { i =>
      if (!lanes.contains(i)) {
        previous(i)
      } else if (accumulate) {
        add(previous(i), input(i), saturate)
      } else {
        input(i)
      }
    }
  }
}


// ============================================================
// RMW tester
// ============================================================

class PoTalAccumulatorRMWTester(
  dut: PoTalAccumulatorRMWHarness,
  saturate: Boolean,
  accLatency: Int
) extends PeekPokeTester(dut) {

  import PoTalRMWGolden._

  val allLanes: Set[Int] = (0 until 4).toSet

  // Initialize test inputs.
  for (bank <- 0 until 2) {
    poke(dut.io.write(bank).valid, 0)
    poke(dut.io.write(bank).bits.addr, 0)
    poke(dut.io.write(bank).bits.acc, 0)

    poke(dut.io.readReq(bank).valid, 0)
    poke(dut.io.readReq(bank).bits, 0)
    poke(dut.io.readResp(bank).ready, 0)

    for (row <- 0 until 2; col <- 0 until 2) {
      poke(dut.io.write(bank).bits.data(row)(col), 0)
    }

    for (byte <- 0 until 16) {
      poke(dut.io.write(bank).bits.mask(byte), 0)
    }
  }

  // ----------------------------------------------------------
  // Write one row. Supports masks and backpressure checking.
  // ----------------------------------------------------------

  def writeRow(
    bank: Int,
    address: Int,
    data: Seq[BigInt],
    accumulate: Boolean,
    lanes: Set[Int] = allLanes,
    mustStall: Boolean = false
  ): Unit = {

    require(data.size == 4)
    require(bank >= 0 && bank < 2)

    val port = dut.io.write(bank)

    poke(port.bits.addr, address)
    poke(port.bits.acc, if (accumulate) 1 else 0)

    for (lane <- 0 until 4) {
      poke(
        port.bits.data(lane / 2)(lane % 2),
        data(lane)
      )
    }

    // Four byte masks per INT32 element.
    for (byte <- 0 until 16) {
      val enabled = lanes.contains(byte / 4)
      poke(port.bits.mask(byte), if (enabled) 1 else 0)
    }

    poke(port.valid, 1)

    if (mustStall) {
      expect(port.ready, 0)
    }

    var waitCycles = 0

    while (peek(port.ready) == 0 && waitCycles < 32) {
      step(1)
      waitCycles += 1
    }

    assert(
      peek(port.ready) == 1,
      s"ACC write timeout: bank=$bank addr=$address"
    )

    // Commit accepted request.
    step(1)
    poke(port.valid, 0)
  }

  // Allow the RMW pipeline and SRAM writes to settle.
  def drain(): Unit = {
    step(accLatency + 4)
  }

  // ----------------------------------------------------------
  // Read one ACC row and check all four INT32 lanes.
  // ----------------------------------------------------------

  def readExpect(
    bank: Int,
    address: Int,
    expected: Seq[BigInt]
  ): Unit = {

    require(expected.size == 4)

    val req = dut.io.readReq(bank)
    val resp = dut.io.readResp(bank)

    poke(resp.ready, 0)
    poke(req.bits, address)
    poke(req.valid, 1)

    var waitReq = 0

    while (peek(req.ready) == 0 && waitReq < 32) {
      step(1)
      waitReq += 1
    }

    assert(
      peek(req.ready) == 1,
      s"ACC read request timeout: bank=$bank"
    )

    step(1)
    poke(req.valid, 0)

    var waitResp = 0

    while (peek(resp.valid) == 0 && waitResp < 32) {
      step(1)
      waitResp += 1
    }

    assert(
      peek(resp.valid) == 1,
      s"ACC read response timeout: bank=$bank"
    )

    for (lane <- 0 until 4) {
      expect(
        resp.bits(lane / 2)(lane % 2),
        expected(lane)
      )
    }

    // Consume the response only after checking it.
    poke(resp.ready, 1)
    step(1)
    poke(resp.ready, 0)
  }

  // ----------------------------------------------------------
  // Test A: Initialize both accumulator banks.
  // ----------------------------------------------------------

  var bank0: Seq[BigInt] = Seq(
    max32, min32, BigInt(100), BigInt(-100)
  )

  var bank1: Seq[BigInt] = Seq(
    BigInt(10), BigInt(-10),
    BigInt(20), BigInt(-20)
  )

  writeRow(0, 3, bank0, accumulate = false)
  writeRow(1, 5, bank1, accumulate = false)

  drain()

  readExpect(0, 3, bank0)
  readExpect(1, 5, bank1)

  // ----------------------------------------------------------
  // Test B: Back-to-back RMW on different banks.
  // ----------------------------------------------------------

  val delta0 = Seq(
    BigInt(1), BigInt(-1),
    BigInt(200), BigInt(40)
  )

  val delta1 = Seq(
    BigInt(-5), BigInt(5),
    BigInt(6), BigInt(-6)
  )

  // No drain between bank 0 and bank 1.
  writeRow(0, 3, delta0, accumulate = true)
  writeRow(1, 5, delta1, accumulate = true)

  bank0 = update(bank0, delta0, true, allLanes, saturate)
  bank1 = update(bank1, delta1, true, allLanes, saturate)

  drain()

  readExpect(0, 3, bank0)
  readExpect(1, 5, bank1)

  // ----------------------------------------------------------
  // Test C: Saturation followed by opposite-sign addition.
  //
  // Sat32: MAX + 1 => MAX; then MAX - 1 => MAX-1
  // Wrap32: MAX + 1 => MIN; then MIN - 1 => MAX
  // ----------------------------------------------------------

  val delta2 = Seq(
    BigInt(-1), BigInt(1),
    BigInt(1000), BigInt(-1000)
  )

  // Update only lanes 0 and 1.
  writeRow(
    0, 3, delta2,
    accumulate = true,
    lanes = Set(0, 1)
  )

  bank0 = update(
    bank0, delta2, true, Set(0, 1), saturate
  )

  drain()
  readExpect(0, 3, bank0)

  // ----------------------------------------------------------
  // Test D: Same-address RAW hazard.
  //
  // Second accumulation must wait for the previous write.
  // ----------------------------------------------------------

  val delta3 = Seq(
    BigInt(2), BigInt(0),
    BigInt(0), BigInt(0)
  )

  val delta4 = Seq(
    BigInt(-3), BigInt(0),
    BigInt(0), BigInt(0)
  )

  writeRow(
    0, 3, delta3,
    accumulate = true,
    lanes = Set(0)
  )

  // Immediate second request to the same address.
  writeRow(
    0, 3, delta4,
    accumulate = true,
    lanes = Set(0),
    mustStall = true
  )

  bank0 = update(
    bank0, delta3, true, Set(0), saturate
  )
  bank0 = update(
    bank0, delta4, true, Set(0), saturate
  )

  drain()
  readExpect(0, 3, bank0)

  // ----------------------------------------------------------
  // Test E: ACC overwrite (acc=false).
  // ----------------------------------------------------------

  val replacement = Seq(
    BigInt(7), BigInt(8),
    BigInt(9), BigInt(10)
  )

  writeRow(
    1, 5, replacement,
    accumulate = false
  )

  bank1 = replacement

  drain()
  readExpect(1, 5, bank1)

  // ----------------------------------------------------------
  // Test F: Masked overwrite.
  // Only lane 2 may change.
  // ----------------------------------------------------------

  val maskedReplacement = Seq(
    BigInt(111), BigInt(222),
    BigInt(333), BigInt(444)
  )

  writeRow(
    1, 5, maskedReplacement,
    accumulate = false,
    lanes = Set(2)
  )

  bank1 = update(
    bank1, maskedReplacement,
    false, Set(2), saturate
  )

  drain()
  readExpect(1, 5, bank1)

  // ----------------------------------------------------------
  // Test G: Deterministic repeated RMW / bank switching.
  // ----------------------------------------------------------

  val random = new scala.util.Random(
    0x50A1L + accLatency + (if (saturate) 100 else 0)
  )

  for (iteration <- 0 until 12) {
    val bank = iteration % 2
    val address = if (bank == 0) 3 else 5

    val delta = Seq.fill(4) {
      BigInt(random.nextInt(8001) - 4000)
    }

    val lanes = (0 until 4).filter { lane =>
      (iteration + lane) % 3 != 0
    }.toSet

    writeRow(
      bank,
      address,
      delta,
      accumulate = true,
      lanes = lanes
    )

    if (bank == 0) {
      bank0 = update(
        bank0, delta, true, lanes, saturate
      )
    } else {
      bank1 = update(
        bank1, delta, true, lanes, saturate
      )
    }

    if (iteration % 4 == 3) {
      drain()
      readExpect(0, 3, bank0)
      readExpect(1, 5, bank1)
    }
  }

  drain()
  readExpect(0, 3, bank0)
  readExpect(1, 5, bank1)
}


// ============================================================
// ScalaTest suite
// ============================================================

class PoTalAccumulatorRMWSpec extends ChiselFlatSpec {

  behavior of "PoTal Accumulator SRAM RMW"

  for {
    saturate <- Seq(false, true)
    accLatency <- Seq(2, 3)
  } {
    val mode = if (saturate) "sat32" else "wrap32"

    it should s"preserve RMW semantics in $mode at ACC latency $accLatency" in {

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-rmw-$mode-$accLatency"
        ),
        () => new PoTalAccumulatorRMWHarness(
          saturate = saturate,
          accLatency = accLatency
        )
      ) { dut =>
        new PoTalAccumulatorRMWTester(
          dut,
          saturate,
          accLatency
        )
      } should be (true)
    }
  }
}

