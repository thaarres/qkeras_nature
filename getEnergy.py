"""Example code to generate weight and MAC sizes in a json file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras
import tensorflow as tf
from qkeras import QActivation
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from qkeras import quantized_bits

def hybrid_model(bits=15, ints=5):
  inputs = tf.keras.Input((16),name="Input")
  x = QDense(64,
    kernel_quantizer = quantized_bits(bits,ints,1),
    bias_quantizer = quantized_bits(bits,ints,1),name="qdense_1")(inputs)
  x = QActivation('quantized_relu({},{})'.format(bits,ints),name="qact_1")(x)
  x = QDense(32,
    kernel_quantizer = quantized_bits(bits,ints,1),
    bias_quantizer = quantized_bits(bits,ints,1),name="qdense_2")(x)
  x = QActivation('quantized_relu({},{})'.format(bits,ints),name="qact_2")(x)
  x = QDense(32,
    kernel_quantizer = quantized_bits(bits,ints,1),
    bias_quantizer = quantized_bits(bits,ints,1),name="qdense_3")(x)
  x = QActivation('quantized_relu({},{})'.format(bits,ints),name="qact_3")(x)
  x = QDense(5,
   kernel_quantizer = quantized_bits(bits,ints,1),
   bias_quantizer = quantized_bits(bits,ints,1),name="qdense_nclasses")(x)
  predictions = tf.keras.layers.Activation('softmax',name="softmax")(x)
  model = tf.keras.Model(inputs, predictions,name='baseline')
  return model

if __name__ == "__main__":
  # input parameters:
  # process: technology process to use in configuration (horowitz, ...)
  # weights_on_memory: whether to store parameters in dram, sram, or fixed
  # activations_on_memory: store activations in dram or sram
  # rd_wr_on_io: whether load data from dram to sram (consider sram as a cache
  #   for dram. If false, we will assume data will be already in SRAM
  # source_quantizers: quantizers for model input
  # is_inference: whether model has been trained already, which is
  #   needed to compute tighter bounds for QBatchNormalization Power estimation.
  # reference_internal: size to use for weight/bias/activation in
  #   get_reference energy calculation (int8, fp16, fp32)
  # reference_accumulator: accumulator and multiplier type in get_reference
  #   energy calculation
  model = hybrid_model()
  model.summary()

  reference_internal = "fp16"
  reference_accumulator = "fp16"

  # By setting for_reference=True, we create QTools object which uses
  # keras_quantizer to quantize weights/bias and
  # keras_accumulator to quantize MAC variables for all layers. Obviously, this
  # overwrites any quantizers that user specified in the qkeras layers. The
  # purpose of doing so is to enable user to calculate a baseline energy number
  # for a given model architecture and compare it against quantized models.
  q = run_qtools.QTools(
      model,
      # energy calculation using a given process
      process="horowitz",
      # quantizers for model input
      source_quantizers=[quantizers.quantized_bits(15, 5, 1)],
      is_inference=False,
      # absolute path (including filename) of the model weights
      weights_path=None,
      # keras_quantizer to quantize weight/bias in un-quantized keras layers
      keras_quantizer=reference_internal,
      # keras_quantizer to quantize MAC in un-quantized keras layers
      keras_accumulator=reference_accumulator,
      # whether calculate baseline energy
      for_reference=True)

  # caculate energy of the derived data type map.
  ref_energy_dict = q.pe(
      # whether to store parameters in dram, sram, or fixed
      weights_on_memory="fixed",
      # store activations in dram or sram
      activations_on_memory="fixed",
      # minimum sram size in number of bits
      min_sram_size=8*16*1024*1024,
      # whether load data from dram to sram (consider sram as a cache
      # for dram. If false, we will assume data will be already in SRAM
      rd_wr_on_io=False)

  # get stats of energy distribution in each layer
  reference_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, ref_energy_dict)
  # extract sum of energy of each layer according to the rule specified in
  # qtools_settings.cfg.include_energy
  total_reference_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, ref_energy_dict)
  print("Baseline energy profile:", reference_energy_profile)
  print("Total baseline energy:", total_reference_energy)

  # By setting for_reference=False, we quantize the model using quantizers
  # specified by users in qkeras layers. For hybrid models where there are
  # mixture of unquantized keras layers and quantized qkeras layers, we use
  # keras_quantizer to quantize weights/bias and keras_accumulator to quantize
  # MAC variables for all keras layers.
  q = run_qtools.QTools(
      model, process="horowitz",
      source_quantizers=[quantizers.quantized_bits(15, 5, 1)],
      is_inference=True, weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=False)
  trial_energy_dict = q.pe(
      weights_on_memory="fixed",
      activations_on_memory="fixed",
      min_sram_size=8*16*1024*1024,
      rd_wr_on_io=True)
  trial_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  total_trial_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  print("energy profile:", trial_energy_profile)
  print("Total energy = {} picoJoule".format(total_trial_energy))