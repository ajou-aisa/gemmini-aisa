
package gemmini

import chisel3._
import chisel3.iotesters._

class PoTalScaleMemoryTester(
  dut: PoTalScaleMemory,
  entries: Int,
  generationWidth: Int
) extends PeekPokeTester(dut) {

  private val zeroCarrier = BigInt("80000000", 16)
  private val lastAddress = entries - 1
  private val lastGeneration = (1 << generationWidth) - 1

  // ----------------------------------------------------------
  // Initialize all inputs
  // ----------------------------------------------------------

  poke(dut.io.write.valid, 0)
  poke(dut.io.write.bits.address, 0)
  poke(dut.io.write.bits.generation, 0)
  poke(dut.io.write.bits.carrier, 0)

  poke(dut.io.release.valid, 0)
  poke(dut.io.release.bits.address, 0)
  poke(dut.io.release.bits.generation, 0)

  poke(dut.io.lookup.address, 0)
  poke(dut.io.lookup.generation, 0)

  reset(2)

  // ----------------------------------------------------------
  // Lookup and check metadata
  // ----------------------------------------------------------

  private def checkLookup(
    address: Int,
    generation: Int,
    hit: Boolean,
    expectedCarrier: BigInt = BigInt(0),
    expectedShift: Int = 0,
    expectedZero: Boolean = false
  ): Unit = {

    poke(dut.io.lookup.address, address)
    poke(dut.io.lookup.generation, generation)

    expect(dut.io.hit, if (hit) 1 else 0)
    expect(dut.io.carrier, expectedCarrier)

    expect(dut.io.metadata.shift, expectedShift)
    expect(
      dut.io.metadata.zero,
      if (expectedZero) 1 else 0
    )

    expect(
      dut.io.metadata.generation,
      if (hit) generation else 0
    )
  }

  // ----------------------------------------------------------
  // Attempt one scale allocation
  // ----------------------------------------------------------

  private def write(
    address: Int,
    generation: Int,
    carrier: BigInt,
    accepted: Boolean,
    invalid: Boolean = false,
    conflict: Boolean = false
  ): Unit = {

    poke(dut.io.write.bits.address, address)
    poke(dut.io.write.bits.generation, generation)
    poke(dut.io.write.bits.carrier, carrier)
    poke(dut.io.write.valid, 1)

    expect(
      dut.io.write.ready,
      if (accepted) 1 else 0
    )

    expect(
      dut.io.invalidWrite,
      if (invalid) 1 else 0
    )

    expect(
      dut.io.writeConflict,
      if (conflict) 1 else 0
    )

    // If ready=0, no write is committed.
    step(1)
    poke(dut.io.write.valid, 0)

    expect(dut.io.invalidWrite, 0)
    expect(dut.io.writeConflict, 0)
  }

  // ----------------------------------------------------------
  // Release one scale entry
  // ----------------------------------------------------------

  private def release(
    address: Int,
    generation: Int,
    shouldMiss: Boolean = false
  ): Unit = {

    poke(dut.io.release.bits.address, address)
    poke(dut.io.release.bits.generation, generation)
    poke(dut.io.release.valid, 1)

    expect(
      dut.io.releaseMiss,
      if (shouldMiss) 1 else 0
    )

    step(1)
    poke(dut.io.release.valid, 0)

    expect(dut.io.releaseMiss, 0)
  }

  // ==========================================================
  // Test A: Empty entry must miss
  // ==========================================================

  checkLookup(
    address = 1,
    generation = 3,
    hit = false
  )

  // ==========================================================
  // Test B: Allocate shift metadata
  // ==========================================================

  write(
    address = 1,
    generation = 3,
    carrier = 7,
    accepted = true
  )

  checkLookup(
    address = 1,
    generation = 3,
    hit = true,
    expectedCarrier = 7,
    expectedShift = 7
  )

  // Wrong generation cannot read an occupied entry.
  checkLookup(
    address = 1,
    generation = 4,
    hit = false
  )

  // ==========================================================
  // Test C: Occupied entry is immutable
  // ==========================================================

  write(
    address = 1,
    generation = 4,
    carrier = 9,
    accepted = false,
    conflict = true
  )

  checkLookup(
    address = 1,
    generation = 3,
    hit = true,
    expectedCarrier = 7,
    expectedShift = 7
  )

  // ==========================================================
  // Test D: Stale release must not invalidate the entry
  // ==========================================================

  release(
    address = 1,
    generation = 2,
    shouldMiss = true
  )

  checkLookup(
    address = 1,
    generation = 3,
    hit = true,
    expectedCarrier = 7,
    expectedShift = 7
  )

  // Correct generation releases the entry.
  release(address = 1, generation = 3)

  checkLookup(
    address = 1,
    generation = 3,
    hit = false
  )

  // ==========================================================
  // Test E: Zero carrier and exponent zero differ
  // ==========================================================

  write(
    address = 1,
    generation = 4,
    carrier = zeroCarrier,
    accepted = true
  )

  checkLookup(
    address = 1,
    generation = 4,
    hit = true,
    expectedCarrier = zeroCarrier,
    expectedShift = 0,
    expectedZero = true
  )

  release(address = 1, generation = 4)

  write(
    address = 1,
    generation = 5,
    carrier = 0,
    accepted = true
  )

  checkLookup(
    address = 1,
    generation = 5,
    hit = true,
    expectedCarrier = 0,
    expectedShift = 0,
    expectedZero = false
  )

  release(address = 1, generation = 5)

  // ==========================================================
  // Test F: Maximum valid exponent
  // ==========================================================

  write(
    address = 1,
    generation = 6,
    carrier = 32767,
    accepted = true
  )

  checkLookup(
    address = 1,
    generation = 6,
    hit = true,
    expectedCarrier = 32767,
    expectedShift = 32767
  )

  release(address = 1, generation = 6)

  // ==========================================================
  // Test G: Invalid carriers must never allocate entries
  // ==========================================================

  val invalidCarriers = Seq(
    BigInt(32768),
    BigInt("80000001", 16),
    BigInt("ffffffff", 16)
  )

  for (carrier <- invalidCarriers) {
    write(
      address = 1,
      generation = 7,
      carrier = carrier,
      accepted = false,
      invalid = true
    )

    checkLookup(
      address = 1,
      generation = 7,
      hit = false
    )
  }

  // ==========================================================
  // Test H: Independent addresses
  // ==========================================================

  write(
    address = 0,
    generation = 1,
    carrier = 2,
    accepted = true
  )

  write(
    address = lastAddress,
    generation = 2,
    carrier = 3,
    accepted = true
  )

  checkLookup(
    address = 0,
    generation = 1,
    hit = true,
    expectedCarrier = 2,
    expectedShift = 2
  )

  checkLookup(
    address = lastAddress,
    generation = 2,
    hit = true,
    expectedCarrier = 3,
    expectedShift = 3
  )

  release(address = 0, generation = 1)

  // Releasing address 0 must not affect another entry.
  checkLookup(
    address = lastAddress,
    generation = 2,
    hit = true,
    expectedCarrier = 3,
    expectedShift = 3
  )

  write(
    address = 0,
    generation = 9,
    carrier = 11,
    accepted = true
  )

  // ==========================================================
  // Test I: Concurrent write and release,
  //         using different addresses
  // ==========================================================

  poke(dut.io.write.bits.address, 2)
  poke(dut.io.write.bits.generation, 10)
  poke(dut.io.write.bits.carrier, 4)
  poke(dut.io.write.valid, 1)

  poke(dut.io.release.bits.address, lastAddress)
  poke(dut.io.release.bits.generation, 2)
  poke(dut.io.release.valid, 1)

  expect(dut.io.write.ready, 1)
  expect(dut.io.writeConflict, 0)
  expect(dut.io.invalidWrite, 0)
  expect(dut.io.releaseMiss, 0)

  step(1)

  poke(dut.io.write.valid, 0)
  poke(dut.io.release.valid, 0)

  checkLookup(
    address = 2,
    generation = 10,
    hit = true,
    expectedCarrier = 4,
    expectedShift = 4
  )

  checkLookup(
    address = lastAddress,
    generation = 2,
    hit = false
  )

  // ==========================================================
  // Test J: Release and conflicting write to same address
  //
  // The occupied entry cannot be replaced until the
  // matching release has taken effect.
  // ==========================================================

  poke(dut.io.write.bits.address, 0)
  poke(dut.io.write.bits.generation, 11)
  poke(dut.io.write.bits.carrier, 6)
  poke(dut.io.write.valid, 1)

  poke(dut.io.release.bits.address, 0)
  poke(dut.io.release.bits.generation, 9)
  poke(dut.io.release.valid, 1)

  expect(dut.io.write.ready, 0)
  expect(dut.io.writeConflict, 1)
  expect(dut.io.releaseMiss, 0)

  step(1)

  poke(dut.io.write.valid, 0)
  poke(dut.io.release.valid, 0)

  checkLookup(
    address = 0,
    generation = 9,
    hit = false
  )

  // The next cycle may allocate the released address.
  write(
    address = 0,
    generation = 11,
    carrier = 6,
    accepted = true
  )

  checkLookup(
    address = 0,
    generation = 11,
    hit = true,
    expectedCarrier = 6,
    expectedShift = 6
  )

  // ==========================================================
  // Test K: Generation reuse across a wrap boundary
  //
  // Reuse is legal here because the preceding generation
  // has already been released.
  // ==========================================================

  release(address = 0, generation = 11)

  write(
    address = 0,
    generation = lastGeneration,
    carrier = 13,
    accepted = true
  )

  checkLookup(
    address = 0,
    generation = lastGeneration,
    hit = true,
    expectedCarrier = 13,
    expectedShift = 13
  )

  release(address = 0, generation = lastGeneration)

  write(
    address = 0,
    generation = 0,
    carrier = 14,
    accepted = true
  )

  checkLookup(
    address = 0,
    generation = lastGeneration,
    hit = false
  )

  checkLookup(
    address = 0,
    generation = 0,
    hit = true,
    expectedCarrier = 14,
    expectedShift = 14
  )
}


// ============================================================
// ScalaTest
// ============================================================

class PoTalScaleMemorySpec extends ChiselFlatSpec {

  behavior of "PoTalScaleMemory"

  for {
    entries <- Seq(4, 8)
    generationWidth <- Seq(4, 8)
  } {

    it should
      s"enforce scale ownership with $entries entries and generation width $generationWidth" in {

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-scale-$entries-$generationWidth"
        ),
        () => new PoTalScaleMemory(
          entries = entries,
          generationWidth = generationWidth
        )
      ) { dut =>
        new PoTalScaleMemoryTester(
          dut,
          entries,
          generationWidth
        )
      } should be (true)
    }
  }
}

