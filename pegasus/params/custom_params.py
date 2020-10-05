# Copyright 2020 The PEGASUS Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarization params of baseline models for downstream datasets."""
from pegasus.params.public_params import transformer_params
from pegasus.params import registry

@registry.register("bioasq_transformer")
def bioasq_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:bioasq/multi_doc-train",
          "dev_pattern": "tfds_transformed:bioasq/multi_doc-validation",
          "test_pattern": "tfds_transformed:bioasq/multi_doc-test",
          "max_input_len": 768,
          "max_output_len": 140,
          "min_output_len": 55,
          "train_steps": 1000,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("mediqa_transformer")
def mediqa_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:mediqa/section2answer_multi_abstractive-train",
          "dev_pattern": "tfds_transformed:mediqa/section2answer_multi_abstractive-validation",
          "test_pattern": "tfds_transformed:mediqa/section2answer_multi_abstractive-test",
          "max_input_len": 768,
          "max_output_len": 140,
          "min_output_len": 55,
          "train_steps": 100,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("duc_2007_transformer")
def duc_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:duc/2007-train",
          "dev_pattern": "tfds_transformed:duc/2007-validation",
          "test_pattern": "tfds_transformed:duc/2007-test",
          "max_input_len": 768,
          "max_output_len": 300,
          "min_output_len": 140,
          "train_steps": 100,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("tac_2009_transformer")
def tac_2009_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:tac/2009-train",
          "dev_pattern": "tfds_transformed:tac/2009-validation",
          "test_pattern": "tfds_transformed:tac/2009-test",
          "max_input_len": 768,
          "max_output_len": 140,
          "min_output_len": 55,
          "train_steps": 100,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("tac_2010_transformer")
def tac_2010_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:tac/2010-train",
          "dev_pattern": "tfds_transformed:tac/2010-validation",
          "test_pattern": "tfds_transformed:tac/2010-test",
          "max_input_len": 768,
          "max_output_len": 140,
          "min_output_len": 55,
          "train_steps": 100,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)
