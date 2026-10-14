package gemmini

import chisel3._
import chisel3.iotesters._

class PoTalWritebackTester(
  dut: PoTalWriteback[SInt, Float, Float],
  scenario: Int
) extends PeekPokeTester(dut) {
  private val zero = BigInt("80000000", 16)
  private val dim = 16

  for (bank <- 0 until 2) {
    poke(dut.io.rawWrite(bank).valid, 0)
    poke(dut.io.scaledWrite(bank).ready, 0)
    poke(dut.io.rawWrite(bank).bits.addr, 0)
    poke(dut.io.rawWrite(bank).bits.acc, 0)
    for (column <- 0 until dim) {
      poke(dut.io.rawWrite(bank).bits.data(column)(0), 0)
    }
    for (byte <- 0 until 4 * dim) {
      poke(dut.io.rawWrite(bank).bits.mask(byte), 0)
    }
  }
  poke(dut.io.contextIssue.valid, 0)
  poke(dut.io.scaleLoad.valid, 0)
  poke(dut.io.scaleRelease.valid, 0)
  poke(dut.io.rawCompletion.valid, 0)
  poke(dut.io.rawCompletion.bits, 0)
  poke(dut.io.robCompletion.ready, 0)
  poke(dut.io.workDone.ready, 0)
  reset(2)

  def load(col: Int, slot: Int, carrier: BigInt, gen: Int = 1): Unit = {
    poke(dut.io.scaleLoad.bits.column, col)
    poke(dut.io.scaleLoad.bits.address, slot)
    poke(dut.io.scaleLoad.bits.generation, gen)
    poke(dut.io.scaleLoad.bits.carrier, carrier)
    poke(dut.io.scaleLoad.valid, 1)
    expect(dut.io.scaleLoad.ready, 1)
    step(1)
    poke(dut.io.scaleLoad.valid, 0)
  }

  def release(col: Int, slot: Int, gen: Int = 1): Unit = {
    poke(dut.io.scaleRelease.bits.column, col)
    poke(dut.io.scaleRelease.bits.address, slot)
    poke(dut.io.scaleRelease.bits.generation, gen)
    poke(dut.io.scaleRelease.valid, 1)
    expect(dut.io.scaleRelease.ready, 1)
    step(1)
    poke(dut.io.scaleRelease.valid, 0)
  }

  def issue(
    rob: Int,
    work: Int,
    block: Int,
    first: Boolean,
    last: Boolean,
    rows: Int = 1,
    cols: Int = 2,
    base: Int = 0,
    nTiles: Int = 1,
    tile: Int = 0,
    gen: Int = 1,
    accepted: Boolean = true
  ): Unit = {
    val ctx = dut.io.contextIssue
    poke(ctx.bits.robId, rob)
    poke(ctx.bits.workId, work)
    poke(ctx.bits.blockIndex, block)
    poke(ctx.bits.scaleBase, base)
    poke(ctx.bits.outputTileCount, nTiles)
    poke(ctx.bits.outputTileIndex, tile)
    poke(ctx.bits.scaleGeneration, gen)
    poke(ctx.bits.validRows, rows)
    poke(ctx.bits.validColumns, cols)
    poke(ctx.bits.firstContribution, if (first) 1 else 0)
    poke(ctx.bits.lastContribution, if (last) 1 else 0)
    poke(ctx.valid, 1)
    expect(ctx.ready, if (accepted) 1 else 0)
    if (accepted) step(1)
    poke(ctx.valid, 0)
  }

  def raw(bank: Int, a: BigInt, b: BigInt,
          accumulate: Boolean, row: Int = 7, cols: Int = 2): Unit = {
    val port = dut.io.rawWrite(bank)
    poke(port.bits.addr, row)
    poke(port.bits.acc, if (accumulate) 1 else 0)
    for (column <- 0 until dim) {
      val value = if (column == 0) a else if (column == 1) b else BigInt(0)
      poke(port.bits.data(column)(0), value)
    }
    for (byte <- 0 until 4 * dim) {
      poke(port.bits.mask(byte), if (byte < 4 * cols) 1 else 0)
    }
    poke(port.valid, 1)
    expect(port.ready, 1)
    step(1)
    poke(port.valid, 0)
  }

  def rawComplete(rob: Int): Unit = {
    poke(dut.io.rawCompletion.bits, rob)
    poke(dut.io.rawCompletion.valid, 1)
    step(1)
    poke(dut.io.rawCompletion.valid, 0)
  }

  if (scenario == 0) {
    // Two passes in K32 block 0 use the same column carriers;
    // block 1 uses a different slot and a zero carrier.
    load(0, 0, 2)
    load(1, 0, 1)
    load(0, 1, 1)
    load(1, 1, zero)

    issue(rob=1, work=0, block=0, first=true, last=false)
    raw(bank=0, a = 3, b = -5, accumulate = false)
    rawComplete(1)

    issue(rob=2, work=0, block=0, first=false, last=false)
    raw(bank=0, a=4, b=9, accumulate=true)
    rawComplete(2)

    issue(rob=3, work=0, block=1, first=false, last=true)
    raw(bank=0, a = 7, b = -100, accumulate=true)
    rawComplete(3)

    // No ACC acceptance -> no output publication or ROB completion.
    step(4)
    expect(dut.io.robCompletion.valid, 0)
    expect(dut.io.workDone.valid, 0)

    poke(dut.io.scaledWrite(0).ready, 1)
    poke(dut.io.robCompletion.ready, 1)
    poke(dut.io.workDone.ready, 1)

    val expected = Seq(
      (BigInt(12), BigInt(-10), false),
      (BigInt(16), BigInt(18), true),
      (BigInt(14), BigInt(0), true)
    )
    var rowCount = 0
    val completedRob = scala.collection.mutable.ArrayBuffer.empty[Int]
    val completedWork = scala.collection.mutable.ArrayBuffer.empty[Int]

    for (_ <- 0 until 50) {
      if (peek(dut.io.scaledWrite(0).valid) == 1) {
        assert(rowCount < expected.size)
        val (a, b, acc) = expected(rowCount)
        expect(dut.io.scaledWrite(0).bits.data(0)(0), a)
        expect(dut.io.scaledWrite(0).bits.data(1)(0), b)
        expect(dut.io.scaledWrite(0).bits.acc, if (acc) 1 else 0)
        expect(dut.io.scaledWrite(0).bits.addr, 7)
        expect(dut.io.scaledWrite(0).bits.mask(0), 1)
        expect(dut.io.scaledWrite(0).bits.mask(8), 0)
        rowCount += 1
      }
      if (peek(dut.io.robCompletion.valid) == 1) {
        completedRob += peek(dut.io.robCompletion.bits).toInt
      }
      if (peek(dut.io.workDone.valid) == 1) {
        completedWork += peek(dut.io.workDone.bits).toInt
      }
      step(1)
    }
    assert(rowCount == 3)
    assert(completedRob.sorted.toSeq == Seq(1,2,3))
    assert(completedWork.toSeq == Seq(0))
    expect(dut.io.drained, 1)

  } else if (scenario == 1) {
    // The scale must be captured when raw output arrives, not
    // re-read when the delayed ACC write finally takes place.
    load(0, 0, 2)
    load(1, 0, 1)
    issue(rob=1, work=0, block=0, first=true, last=true)
    raw(bank=0, a=3, b = -5, accumulate=false)
    rawComplete(1)

    release(0, 0)
    release(1, 0)
    load(0, 0, 0, gen=2)
    load(1, 0, zero, gen=2)

    issue(rob=2, work=1, block=0,
      first=true, last=true, gen=1, accepted=false)
    expect(dut.io.error.scaleMiss, 0) // valid is now low

    poke(dut.io.scaledWrite(0).ready, 1)
    poke(dut.io.robCompletion.ready, 1)
    poke(dut.io.workDone.ready, 1)
    var rowCount = 0
    val robs = scala.collection.mutable.ArrayBuffer.empty[Int]
    for (_ <- 0 until 20) {
      if (peek(dut.io.scaledWrite(0).valid) == 1) {
        expect(dut.io.scaledWrite(0).bits.data(0)(0), 12)
        expect(dut.io.scaledWrite(0).bits.data(1)(0), -10)
        rowCount += 1
      }
      if (peek(dut.io.robCompletion.valid) == 1)
        robs += peek(dut.io.robCompletion.bits).toInt
      step(1)
    }
    assert(rowCount == 1)
    assert(robs.toSeq == Seq(1))

  } else if (scenario == 2) {
    // Exact cache slot = base + block * outputTileCount + tile.
    load(0, 3, 3)
    issue(rob=1, work=0, block=1, first=true, last=true,
      cols=1, base=0, nTiles=2, tile=1)
    raw(bank=1, a=6, b=0, accumulate=false, cols=1)
    rawComplete(1)
    poke(dut.io.scaledWrite(1).ready, 1)
    poke(dut.io.robCompletion.ready, 1)
    poke(dut.io.workDone.ready, 1)
    var rows = 0
    for (_ <- 0 until 20) {
      if (peek(dut.io.scaledWrite(1).valid) == 1) {
        expect(dut.io.scaledWrite(1).bits.data(0)(0), 48)
        expect(dut.io.scaledWrite(1).bits.mask(0), 1)
        expect(dut.io.scaledWrite(1).bits.mask(4), 0)
        rows += 1
      }
      step(1)
    }
    assert(rows == 1)
    expect(dut.io.drained, 1)

  } else if (scenario == 3) {
    // All row credits consumed by the first accepted context.
    load(0, 0, 0)
    issue(rob=1, work=0, block=0,
      first=true, last=false, rows=dim, cols=1)
    issue(rob=2, work=1, block=0,
      first=true, last=true, rows=1, cols=1, accepted=false)
    expect(dut.io.pipelineDrained, 0)
  } else if (scenario == 4) {
    // W3-2c-9a:
    // Context queue full + row-credit exhaustion +
    // automatic admission recovery.
    //
    // contextDepth = 2
    // queueDepth   = 32
    // validRows    = 16 per context

    // Three independent scale slots.
    load(0, 0, 0) // x1
    load(0, 1, 1) // x2
    load(0, 2, 2) // x4

    // Reserve all 32 row credits.
    issue(
      rob=1, work=0, block=0,
      first=true, last=true,
      rows=dim, cols=1
    )
    issue(
      rob=2, work=1, block=1,
      first=true, last=true,
      rows=dim, cols=1
    )

    // Third context must not be accepted.
    issue(
      rob=3, work=2, block=2,
      first=true, last=true,
      rows=dim, cols=1,
      accepted=false
    )

    // Keep context 3 pending, as a stalled PRELOAD would.
    val third = dut.io.contextIssue
    poke(third.valid, 1)

    for (_ <- 0 until 4) {
      expect(third.ready, 0)
      expect(dut.io.pipelineDrained, 0)
      step(1)
    }

    // Context 1 produces all 16 raw rows.
    // Physical ACC writes remain blocked.
    for (row <- 0 until dim) {
      expect(third.ready, 0)
      raw(
        bank=0,
        a=BigInt(row + 1),
        b=BigInt(0),
        accumulate=false,
        row=row,
        cols=1
      )
    }
    rawComplete(1)

    // Context 1 has left the context FIFO,
    // but no row credits have been returned.
    expect(third.ready, 0)
    expect(dut.io.robCompletion.valid, 0)
    expect(dut.io.workDone.valid, 0)

    // Drain exactly 16 physical ACC writes.
    poke(dut.io.scaledWrite(0).ready, 1)

    for (row <- 0 until dim) {
      // Until all 16 credits return, context 3
      // must remain blocked.
      expect(third.ready, 0)
      expect(dut.io.scaledWrite(0).valid, 1)
      expect(dut.io.scaledWrite(1).valid, 0)

      expect(
        dut.io.scaledWrite(0).bits.addr, row
      )
      expect(
        dut.io.scaledWrite(0).bits.data(0)(0),
        row + 1
      )
      expect(
        dut.io.scaledWrite(0).bits.acc, 0
      )
      expect(
        dut.io.scaledWrite(0).bits.mask(0), 1
      )
      expect(
        dut.io.scaledWrite(0).bits.mask(4), 0
      )

      step(1)
    }

    // 16 credits have been returned.
    // The pending third context must now be accepted.
    expect(third.ready, 1)
    step(1)
    poke(third.valid, 0)

    // Block physical ACC writes again.
    poke(dut.io.scaledWrite(0).ready, 0)

    // Context 2: raw=2*(row+1), scale=x2.
    for (row <- 0 until dim) {
      raw(
        bank=0,
        a=BigInt(2 * (row + 1)),
        b=BigInt(0),
        accumulate=false,
        row=row,
        cols=1
      )
    }
    rawComplete(2)

    // Context 3: raw=3*(row+1), scale=x4.
    for (row <- 0 until dim) {
      raw(
        bank=0,
        a=BigInt(3 * (row + 1)),
        b=BigInt(0),
        accumulate=false,
        row=row,
        cols=1
      )
    }
    rawComplete(3)

    // No further writes accepted while ready=0.
    expect(dut.io.pipelineDrained, 0)

    // Drain the remaining 32 rows.
    poke(dut.io.scaledWrite(0).ready, 1)

    for (index <- 0 until 2 * dim) {
      val row = index % dim

      val expected =
        if (index < dim) 4 * (row + 1)
        else 12 * (row + 1)

      expect(dut.io.scaledWrite(0).valid, 1)
      expect(dut.io.scaledWrite(1).valid, 0)
      expect(dut.io.scaledWrite(0).bits.addr, row)
      expect(
        dut.io.scaledWrite(0).bits.data(0)(0),
        expected
      )
      expect(dut.io.scaledWrite(0).bits.acc, 0)
      expect(dut.io.scaledWrite(0).bits.mask(0), 1)

      step(1)
    }

    // Publish completed ROB IDs and work IDs.
    poke(dut.io.robCompletion.ready, 1)
    poke(dut.io.workDone.ready, 1)

    val robs =
      scala.collection.mutable.ArrayBuffer.empty[Int]
    val works =
      scala.collection.mutable.ArrayBuffer.empty[Int]

    for (_ <- 0 until 50) {
      if (peek(dut.io.robCompletion.valid) == 1) {
        robs += peek(dut.io.robCompletion.bits).toInt
      }

      if (peek(dut.io.workDone.valid) == 1) {
        works += peek(dut.io.workDone.bits).toInt
      }

      step(1)
    }

    assert(
      robs.sorted.toSeq == Seq(1, 2, 3),
      s"Unexpected ROB completions: $robs"
    )

    assert(
      works.sorted.toSeq == Seq(0, 1, 2),
      s"Unexpected work completions: $works"
    )

    expect(dut.io.pipelineDrained, 1)
    expect(dut.io.drained, 1)

    // Scale entries can be released after drain.
    release(0, 0)
    release(0, 1)
    release(0, 2)

  } else {
    throw new IllegalArgumentException("unknown scenario")
  }
}

class PoTalWritebackSpec extends ChiselFlatSpec {
  behavior of "PoTalWriteback"

  for (scenario <- 0 until 5) {
    it should s"preserve K32 scale, row credit and completion scenario $scenario" in {
      val cfg = GemminiConfigs.defaultConfig.copy(
        dataflow = Dataflow.WS,
        reservation_station_entries_ex = 2,
        acc_latency = 2)
      val depth = if (scenario == 3) 16 else 32

      Driver.execute(
        Array("--backend-name", "treadle", "--target-dir",
          s"test_run_dir/potal-writeback-$scenario"),
        () => new PoTalWriteback(cfg,
          scaleEntries=4, generationWidth=4, workEntries=4,
          contextDepth=2, queueDepth=depth)
      ) { dut => new PoTalWritebackTester(dut, scenario) } should be (true)
    }
  }
}

