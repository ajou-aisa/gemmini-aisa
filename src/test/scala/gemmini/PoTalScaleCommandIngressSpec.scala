
package gemmini

import chisel3._
import chisel3.util._
import chisel3.iotesters._

// ============================================================
// Ingress + actual PoTalScaleMemory banks
//
// The production design has 256 slots per column.
// The unit test uses 8 slots to reduce simulation time.
// ============================================================

final class PoTalScaleIngressHarness(
  dim: Int,
  entries: Int = 8
) extends Module {

  private val slotWidth = log2Ceil(entries)
  private val columnWidth = log2Ceil(dim)

  val io = IO(new Bundle {
    val cmd = Flipped(
      Decoupled(new PoTalScaleCommand)
    )

    val resp = Decoupled(
      new PoTalScaleCommandResponse
    )

    val releaseSafe = Input(Bool())
    val fenceDrained = Input(Bool())

    val lookupColumn = Input(UInt(6.W))
    val lookupAddress = Input(UInt(8.W))
    val lookupGeneration = Input(UInt(8.W))

    val lookupHit = Output(Bool())
    val lookupCarrier = Output(UInt(32.W))
  })

  val ingress = Module(
    new PoTalScaleCommandIngress(
      dim = dim,
      scaleEntries = entries,
      generationWidth = 8
    )
  )

  val memories = Seq.fill(dim) {
    Module(new PoTalScaleMemory(entries, 8))
  }

  // External command and response.
  io.cmd <> ingress.io.cmd
  io.resp <> ingress.io.resp

  ingress.io.releaseSafe := io.releaseSafe
  ingress.io.fenceDrained := io.fenceDrained

  // Match the ScaleMemory wiring currently used
  // inside PoTalWriteback.
  ingress.io.scaleLoad.ready :=
    VecInit(memories.map(_.io.write.ready))(
      ingress.io.scaleLoad.bits.column
    )

  ingress.io.scaleRelease.ready := true.B

  ingress.io.scaleConflict :=
    memories.map(_.io.writeConflict).reduce(_ || _)

  ingress.io.releaseMiss :=
    memories.map(_.io.releaseMiss).reduce(_ || _)

  for ((memory, column) <- memories.zipWithIndex) {

    memory.io.write.valid :=
      ingress.io.scaleLoad.valid &&
      ingress.io.scaleLoad.bits.column === column.U

    memory.io.write.bits.address :=
      ingress.io.scaleLoad.bits.address

    memory.io.write.bits.generation :=
      ingress.io.scaleLoad.bits.generation

    memory.io.write.bits.carrier :=
      ingress.io.scaleLoad.bits.carrier

    memory.io.release.valid :=
      ingress.io.scaleRelease.fire &&
      ingress.io.scaleRelease.bits.column === column.U

    memory.io.release.bits.address :=
      ingress.io.scaleRelease.bits.address

    memory.io.release.bits.generation :=
      ingress.io.scaleRelease.bits.generation

    memory.io.lookup.address :=
      io.lookupAddress(slotWidth - 1, 0)

    memory.io.lookup.generation :=
      io.lookupGeneration
  }

  val selectedColumn =
    io.lookupColumn(columnWidth - 1, 0)

  io.lookupHit := Mux(
    io.lookupColumn < dim.U,
    VecInit(memories.map(_.io.hit))(selectedColumn),
    false.B
  )

  io.lookupCarrier := Mux(
    io.lookupColumn < dim.U,
    VecInit(memories.map(_.io.carrier))(selectedColumn),
    0.U(32.W)
  )
}

// ============================================================
// Functional tester
// ============================================================

final class PoTalScaleIngressTester(
  dut: PoTalScaleIngressHarness,
  dim: Int
) extends PeekPokeTester(dut) {

  private val WRITE = 23
  private val RELEASE = 24
  private val FENCE = 25

  private val OK = 0
  private val INVALID_COLUMN = 1
  private val INVALID_SCALE = 2
  private val SCALE_CONFLICT = 3
  private val RELEASE_MISS = 4
  private val INVALID_ADDRESS = 5
  private val INVALID_ENCODING = 6
  private val BAD_COMMAND = 7

  private val zeroCarrier =
    BigInt("80000000", 16)

  private val lastColumn = dim - 1
  private val rd = 11

  // ----------------------------------------------------------
  // Initial signals
  // ----------------------------------------------------------

  poke(dut.io.cmd.valid, 0)
  poke(dut.io.resp.ready, 0)

  poke(dut.io.releaseSafe, 1)
  poke(dut.io.fenceDrained, 1)

  poke(dut.io.lookupColumn, 0)
  poke(dut.io.lookupAddress, 0)
  poke(dut.io.lookupGeneration, 0)

  reset(3)

  // ----------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------

  private def offer(
    funct: Int,
    slot: Int = 0,
    generation: Int = 0,
    column: Int = 0,
    carrier: BigInt = BigInt(0),
    extraRs1: BigInt = BigInt(0)
  ): Unit = {

    val rs1 =
      BigInt(slot) |
      (BigInt(generation) << 8) |
      (BigInt(column) << 16) |
      extraRs1

    poke(dut.io.cmd.bits.funct, funct)
    poke(dut.io.cmd.bits.rs1, rs1)
    poke(dut.io.cmd.bits.rs2, carrier)
    poke(dut.io.cmd.bits.rd, rd)
    poke(dut.io.cmd.bits.xd, 1)

    poke(dut.io.cmd.valid, 1)
  }

  private def accept(expectedStatus: Int): Unit = {

    expect(dut.io.cmd.ready, 1)
    step(1)

    poke(dut.io.cmd.valid, 0)

    expect(dut.io.resp.valid, 1)
    expect(dut.io.resp.bits.rd, rd)
    expect(dut.io.resp.bits.status, expectedStatus)

    // Hold the response for two cycles.
    // The response must not be lost or overwritten.
    step(2)

    expect(dut.io.resp.valid, 1)
    expect(dut.io.resp.bits.rd, rd)
    expect(dut.io.resp.bits.status, expectedStatus)

    poke(dut.io.resp.ready, 1)
    step(1)
    poke(dut.io.resp.ready, 0)

    expect(dut.io.resp.valid, 0)
  }

  private def lookup(
    column: Int,
    slot: Int,
    generation: Int,
    hit: Int,
    carrier: BigInt = BigInt(0)
  ): Unit = {

    poke(dut.io.lookupColumn, column)
    poke(dut.io.lookupAddress, slot)
    poke(dut.io.lookupGeneration, generation)

    expect(dut.io.lookupHit, hit)

    if (hit == 1) {
      expect(dut.io.lookupCarrier, carrier)
    }
  }

  // ----------------------------------------------------------
  // 1. Normal write and lookup
  // ----------------------------------------------------------

  offer(
    funct = WRITE,
    slot = 3,
    generation = 7,
    column = lastColumn,
    carrier = BigInt(3)
  )

  accept(OK)

  lookup(
    lastColumn, 3, 7,
    hit = 1,
    carrier = BigInt(3)
  )

  // ----------------------------------------------------------
  // 2. Duplicate write must return an error.
  //    Existing entry must remain unchanged.
  // ----------------------------------------------------------

  offer(
    funct = WRITE,
    slot = 3,
    generation = 8,
    column = lastColumn,
    carrier = BigInt(4)
  )

  accept(SCALE_CONFLICT)

  lookup(
    lastColumn, 3, 7,
    hit = 1,
    carrier = BigInt(3)
  )

  lookup(lastColumn, 3, 8, hit = 0)

  // ----------------------------------------------------------
  // 3. Wrong-generation release must fail.
  // ----------------------------------------------------------

  offer(
    funct = RELEASE,
    slot = 3,
    generation = 8,
    column = lastColumn
  )

  accept(RELEASE_MISS)

  lookup(
    lastColumn, 3, 7,
    hit = 1,
    carrier = BigInt(3)
  )

  // ----------------------------------------------------------
  // 4. Release must wait for safe drain.
  // ----------------------------------------------------------

  poke(dut.io.releaseSafe, 0)

  offer(
    funct = RELEASE,
    slot = 3,
    generation = 7,
    column = lastColumn
  )

  for (_ <- 0 until 3) {
    expect(dut.io.cmd.ready, 0)
    expect(dut.io.resp.valid, 0)
    step(1)
  }

  lookup(
    lastColumn, 3, 7,
    hit = 1,
    carrier = BigInt(3)
  )

  poke(dut.io.releaseSafe, 1)
  accept(OK)

  lookup(lastColumn, 3, 7, hit = 0)

  // ----------------------------------------------------------
  // 5. Reallocation with new generation / zero carrier
  // ----------------------------------------------------------

  offer(
    funct = WRITE,
    slot = 3,
    generation = 9,
    column = lastColumn,
    carrier = zeroCarrier
  )

  accept(OK)

  lookup(
    lastColumn, 3, 9,
    hit = 1,
    carrier = zeroCarrier
  )

  lookup(lastColumn, 3, 7, hit = 0)

  // ----------------------------------------------------------
  // 6. Different columns own independent entries.
  // ----------------------------------------------------------

  offer(
    funct = WRITE,
    slot = 3,
    generation = 9,
    column = 0,
    carrier = BigInt(2)
  )

  accept(OK)

  lookup(
    0, 3, 9,
    hit = 1,
    carrier = BigInt(2)
  )

  lookup(
    lastColumn, 3, 9,
    hit = 1,
    carrier = zeroCarrier
  )

  // ----------------------------------------------------------
  // 7. Invalid column
  //
  // At DIM64 all six-bit column IDs are valid,
  // so there is no out-of-range six-bit column.
  // ----------------------------------------------------------

  if (dim < 64) {
    offer(
      funct = WRITE,
      slot = 0,
      generation = 1,
      column = dim,
      carrier = BigInt(1)
    )

    accept(INVALID_COLUMN)

  } else {
    // rs1 bit 22 is reserved.
    offer(
      funct = WRITE,
      slot = 0,
      generation = 1,
      column = 0,
      carrier = BigInt(1),
      extraRs1 = BigInt(1) << 22
    )

    accept(INVALID_ENCODING)
  }

  // ----------------------------------------------------------
  // 8. Invalid address (8-slot test configuration)
  // ----------------------------------------------------------

  offer(
    funct = WRITE,
    slot = 8,
    generation = 1,
    column = 0,
    carrier = BigInt(1)
  )

  accept(INVALID_ADDRESS)

  lookup(0, 0, 1, hit = 0)

  // ----------------------------------------------------------
  // 9. Invalid PoT carrier
  // ----------------------------------------------------------

  offer(
    funct = WRITE,
    slot = 4,
    generation = 1,
    column = 0,
    carrier = BigInt("FFFFFFFF", 16)
  )

  accept(INVALID_SCALE)

  lookup(0, 4, 1, hit = 0)

  // ----------------------------------------------------------
  // 10. Fence waits for drained condition.
  // ----------------------------------------------------------

  poke(dut.io.fenceDrained, 0)

  offer(funct = FENCE)

  for (_ <- 0 until 3) {
    expect(dut.io.cmd.ready, 0)
    expect(dut.io.resp.valid, 0)
    step(1)
  }

  poke(dut.io.fenceDrained, 1)

  accept(OK)

  // ----------------------------------------------------------
  // 11. Unknown command must not be forwarded.
  // ----------------------------------------------------------

  offer(funct = 26)
  accept(BAD_COMMAND)

  // Ensure earlier scale entries survived.
  lookup(
    0, 3, 9,
    hit = 1,
    carrier = BigInt(2)
  )

  lookup(
    lastColumn, 3, 9,
    hit = 1,
    carrier = zeroCarrier
  )
}

// ============================================================
// ScalaTest
// ============================================================

class PoTalScaleCommandIngressSpec extends ChiselFlatSpec {

  behavior of "PoTal Scale Command Ingress"

  for (dim <- Seq(16, 32, 64)) {

    it should s"handle scale commands at DIM=$dim" in {

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-scale-ingress-d$dim"
        ),
        () => new PoTalScaleIngressHarness(dim)
      ) { dut =>
        new PoTalScaleIngressTester(dut, dim)
      } should be (true)
    }
  }
}

