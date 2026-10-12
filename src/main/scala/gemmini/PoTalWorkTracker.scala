package gemmini

import chisel3._
import chisel3.util._

// One work ID denotes one output tile across all K32 blocks.
// Work completion requires the last contribution to be issued
// and all accepted output passes to finish their ACC write latency.
final class PoTalWorkTracker(
  entries: Int,
  countWidth: Int = 16
) extends Module {
  require(entries >= 2 && isPow2(entries))
  require(countWidth > 0)

  private val workWidth = math.max(1, log2Ceil(entries))

  val io = IO(new Bundle {
    val allocate = Flipped(Decoupled(UInt(workWidth.W)))
    val issue = Flipped(Valid(UInt(workWidth.W)))
    val seal = Flipped(Valid(UInt(workWidth.W)))
    val commit = Flipped(Valid(UInt(workWidth.W)))
    val done = Decoupled(UInt(workWidth.W))
    val active = Output(Vec(entries, Bool()))
    val duplicateAllocate = Output(Bool())
    val invalidIssue = Output(Bool())
    val invalidSeal = Output(Bool())
    val commitUnderflow = Output(Bool())
    val issueOverflow = Output(Bool())
  })

  private val active = RegInit(VecInit(Seq.fill(entries)(false.B)))
  private val sealedWork = RegInit(VecInit(Seq.fill(entries)(false.B)))
  private val outstanding = RegInit(
    VecInit(Seq.fill(entries)(0.U(countWidth.W))))

  private val complete = VecInit((0 until entries).map { i =>
    active(i) && sealedWork(i) && outstanding(i) === 0.U
  })
  io.done.valid := complete.asUInt.orR
  io.done.bits := PriorityEncoder(complete.asUInt)

  private val retiring = io.done.fire && io.done.bits === io.allocate.bits
  io.allocate.ready := !active(io.allocate.bits) || retiring
  private val allocFire = io.allocate.fire

  private val issueAllowed =
    (active(io.issue.bits) && !sealedWork(io.issue.bits) &&
      !(io.done.fire && io.done.bits === io.issue.bits)) ||
    (allocFire && io.allocate.bits === io.issue.bits)
  private val sealAllowed =
    (active(io.seal.bits) && !(io.done.fire && io.done.bits === io.seal.bits)) ||
    (allocFire && io.allocate.bits === io.seal.bits)
  private val commitAllowed =
    (active(io.commit.bits) && !(io.done.fire && io.done.bits === io.commit.bits)) ||
    (allocFire && io.allocate.bits === io.commit.bits)
  private val sameCycleIssue =
    io.issue.valid && io.issue.bits === io.commit.bits && issueAllowed

  io.duplicateAllocate := io.allocate.valid && !io.allocate.ready
  io.invalidIssue := io.issue.valid && !issueAllowed
  io.invalidSeal := io.seal.valid && !sealAllowed
  io.commitUnderflow := io.commit.valid && (!commitAllowed ||
    (outstanding(io.commit.bits) === 0.U && !sameCycleIssue))
  io.issueOverflow := io.issue.valid && issueAllowed &&
    outstanding(io.issue.bits).andR &&
    !(io.commit.valid && io.commit.bits === io.issue.bits)

  for (i <- 0 until entries) {
    val doneHere = io.done.fire && io.done.bits === i.U
    val allocateHere = allocFire && io.allocate.bits === i.U
    val issueHere = io.issue.valid && io.issue.bits === i.U &&
      !io.invalidIssue && !io.issueOverflow
    val sealHere = io.seal.valid && io.seal.bits === i.U &&
      !io.invalidSeal
    val commitHere = io.commit.valid && io.commit.bits === i.U &&
      !io.commitUnderflow

    when(doneHere) {
      active(i) := false.B
      sealedWork(i) := false.B
      outstanding(i) := 0.U
    }
    when(allocateHere) {
      active(i) := true.B
      sealedWork(i) := false.B
      outstanding(i) := 0.U
    }
    when(sealHere) {
      sealedWork(i) := true.B
    }
    when(issueHere && !commitHere) {
      outstanding(i) := outstanding(i) + 1.U
    }.elsewhen(commitHere && !issueHere) {
      outstanding(i) := outstanding(i) - 1.U
    }
  }

  io.active := active
}

