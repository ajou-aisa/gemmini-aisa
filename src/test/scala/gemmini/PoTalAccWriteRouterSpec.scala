
package gemmini

import chisel3._
import chisel3.iotesters._

final class PoTalAccWriteRouterTester(
  dut: PoTalAccWriteRouter,
  dim: Int
) extends PeekPokeTester(dut) {

  val banks = 2

  // Initialize all inputs.
  for (bank <- 0 until banks) {
    poke(dut.io.exWrite(bank).valid, 0)
    poke(dut.io.isPoTal(bank), 0)
    poke(dut.io.rawWrite(bank).ready, 1)
    poke(dut.io.scaledWrite(bank).valid, 0)
    poke(dut.io.accWrite(bank).ready, 1)

    poke(dut.io.exWrite(bank).bits.addr, 5 + bank)
    poke(dut.io.exWrite(bank).bits.acc, 0)

    poke(dut.io.scaledWrite(bank).bits.addr, 12 + bank)
    poke(dut.io.scaledWrite(bank).bits.acc, 1)

    for (column <- 0 until dim) {
      poke(
        dut.io.exWrite(bank).bits.data(column)(0),
        100 + bank * 100 + column
      )

      poke(
        dut.io.scaledWrite(bank).bits.data(column)(0),
        1000 + bank * 100 + column
      )
    }

    for (byte <- 0 until 4 * dim) {
      val mask = if (byte < 8) 1 else 0

      poke(dut.io.exWrite(bank).bits.mask(byte), mask)
      poke(dut.io.scaledWrite(bank).bits.mask(byte), mask)
    }
  }

  reset(2)

  // ----------------------------------------------------
  // Test 1: Conventional EX has priority over scaled WB.
  // ----------------------------------------------------

  poke(dut.io.exWrite(0).valid, 1)
  poke(dut.io.scaledWrite(0).valid, 1)

  expect(dut.io.exWrite(0).ready, 1)
  expect(dut.io.rawWrite(0).valid, 0)
  expect(dut.io.scaledWrite(0).ready, 0)

  expect(dut.io.accWrite(0).valid, 1)
  expect(dut.io.accWrite(0).bits.addr, 5)
  expect(dut.io.accWrite(0).bits.data(0)(0), 100)
  expect(dut.io.accWrite(0).bits.acc, 0)

  step(1)

  // ----------------------------------------------------
  // Test 2: PoTal raw output bypasses ACC.
  //
  // ACC is stalled but the reserved raw input is ready.
  // ----------------------------------------------------

  poke(dut.io.isPoTal(0), 1)
  poke(dut.io.accWrite(0).ready, 0)

  expect(dut.io.exWrite(0).ready, 1)
  expect(dut.io.rawWrite(0).valid, 1)
  expect(dut.io.rawWrite(0).bits.addr, 5)
  expect(dut.io.rawWrite(0).bits.data(0)(0), 100)

  // ACC sees the earlier scaled output, not raw EX.
  expect(dut.io.accWrite(0).valid, 1)
  expect(dut.io.accWrite(0).bits.addr, 12)
  expect(dut.io.accWrite(0).bits.data(0)(0), 1000)

  expect(dut.io.scaledWrite(0).ready, 0)

  step(1)

  // ----------------------------------------------------
  // Test 3: Scaled output resumes after ACC is ready.
  // ----------------------------------------------------

  poke(dut.io.exWrite(0).valid, 0)
  poke(dut.io.accWrite(0).ready, 1)

  expect(dut.io.rawWrite(0).valid, 0)
  expect(dut.io.scaledWrite(0).ready, 1)

  expect(dut.io.accWrite(0).valid, 1)
  expect(dut.io.accWrite(0).bits.addr, 12)
  expect(dut.io.accWrite(0).bits.data(0)(0), 1000)
  expect(dut.io.accWrite(0).bits.acc, 1)

  step(1)

  // ----------------------------------------------------
  // Test 4: Independent bank routing.
  //
  // Bank 0: conventional EX
  // Bank 1: PoTal EX and earlier scaled write
  // ----------------------------------------------------

  poke(dut.io.scaledWrite(0).valid, 0)

  poke(dut.io.exWrite(0).valid, 1)
  poke(dut.io.isPoTal(0), 0)

  poke(dut.io.exWrite(1).valid, 1)
  poke(dut.io.isPoTal(1), 1)
  poke(dut.io.scaledWrite(1).valid, 1)

  expect(dut.io.rawWrite(0).valid, 0)
  expect(dut.io.accWrite(0).valid, 1)
  expect(dut.io.accWrite(0).bits.data(0)(0), 100)

  expect(dut.io.rawWrite(1).valid, 1)
  expect(dut.io.rawWrite(1).bits.data(0)(0), 200)

  expect(dut.io.accWrite(1).valid, 1)
  expect(dut.io.accWrite(1).bits.data(0)(0), 1100)

  expect(dut.io.exWrite(0).ready, 1)
  expect(dut.io.exWrite(1).ready, 1)

  step(1)

  // ----------------------------------------------------
  // Test 5: Raw destination backpressure propagates.
  //
  // In the final architecture PoTalWriteback must
  // guarantee readiness through prior reservation.
  // ----------------------------------------------------

  poke(dut.io.exWrite(0).valid, 0)
  poke(dut.io.exWrite(1).valid, 1)
  poke(dut.io.isPoTal(1), 1)
  poke(dut.io.rawWrite(1).ready, 0)

  expect(dut.io.rawWrite(1).valid, 1)
  expect(dut.io.exWrite(1).ready, 0)

  step(1)

  poke(dut.io.exWrite(1).valid, 0)
  poke(dut.io.scaledWrite(1).valid, 0)

  expect(dut.io.rawWrite(1).valid, 0)
}

class PoTalAccWriteRouterSpec extends ChiselFlatSpec {

  behavior of "PoTal ACC write routing"

  for (dim <- Seq(16, 32, 64)) {

    it should s"route raw and scaled writes at DIM=$dim" in {

      Driver.execute(
        Array(
          "--backend-name", "treadle",
          "--target-dir",
          s"test_run_dir/potal-acc-router-d$dim"
        ),
        () => new PoTalAccWriteRouter(
          dim = dim,
          accBanks = 2,
          accBankEntries = 128
        )
      ) { dut =>
        new PoTalAccWriteRouterTester(dut, dim)
      } should be (true)
    }
  }
}

