
package gemmini

import chisel3._
import Arithmetic.SIntArithmetic

// ============================================================
// PoTal hardware configurations
// ============================================================

object PoTalConfigs {

  // W3-1:
  // A8W8 / DIM16 / WS / Sat32 ACC
  //
  // The SCU writeback is not yet connected to Controller.
  // Therefore enablePoTalWriteback remains false.

  val a8w8D16Sat32: GemminiArrayConfig[SInt, Float, Float] =
    GemminiConfigs.defaultConfig.copy(

      // ------------------------------------------------------
      // Datatypes
      // ------------------------------------------------------

      inputType = SInt(8.W),
      spatialArrayOutputType = SInt(20.W),
      accType = SInt(32.W),

      // ------------------------------------------------------
      // Systolic array
      // ------------------------------------------------------

      dataflow = Dataflow.WS,

      tileRows = 1,
      tileColumns = 1,
      meshRows = 16,
      meshColumns = 16,

      // ------------------------------------------------------
      // Scratchpad
      // ------------------------------------------------------

      sp_banks = 4,
      sp_singleported = false,
      sp_capacity = CapacityInKilobytes(256),
      spad_read_delay = 4,

      // ------------------------------------------------------
      // Accumulator
      // ------------------------------------------------------

      acc_banks = 2,
      acc_singleported = false,
      acc_sub_banks = 2,
      acc_capacity = CapacityInKilobytes(64),
      acc_latency = 2,

      acc_read_full_width = true,

      // ------------------------------------------------------
      // DMA
      // ------------------------------------------------------

      dma_maxbytes = 64,
      dma_buswidth = 128,

      // ------------------------------------------------------
      // Disable conventional scaling
      //
      // PoTal will use PoTalSCU before ACC writeback.
      // ------------------------------------------------------

      mvin_scale_args = None,
      mvin_scale_acc_args = None,
      mvin_scale_shared = false,
      acc_scale_args = None,

      // ------------------------------------------------------
      // Execution
      // ------------------------------------------------------

      ex_read_from_spad = true,
      ex_read_from_acc = false,
      ex_write_to_spad = false,
      ex_write_to_acc = true,

      hardcode_d_to_garbage_addr = true,

      // ------------------------------------------------------
      // Disable unused DNN functions
      // ------------------------------------------------------

      has_training_convs = false,
      has_max_pool = false,
      has_nonlinear_activations = false,
      has_dw_convs = false,
      has_normalizations = false,
      has_first_layer_optimizations = false,

      // ------------------------------------------------------
      // PoTal features
      // ------------------------------------------------------

      enablePoTalSat32Acc = true,

      // Will be enabled after W3-2 integration.
      enablePoTalWriteback = false,

      // Keep the default Gemmini header untouched.
      headerFileName = "gemmini_potal_params.h"
    )
}


// ============================================================
// Rocket RoCC Config
//
// Reuse the existing Gemmini LazyRoCC implementation.
// No alternate accelerator top is introduced.
// ============================================================

class PoTalSat32GemminiConfig
    extends DefaultGemminiConfig[SInt, Float, Float](
      PoTalConfigs.a8w8D16Sat32
    )

