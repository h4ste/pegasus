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

# Lint as: python3
"""Public Supervised Datasets.

Supervised datasets for finetuning, available through public TFDS.
A supervised dataset provides (input, output) tuple when created with
as_supervised option.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from pegasus.data import datasets
from pegasus.data.public_supervised_datasets import PublicSupervisedTFDSDataset

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('fsl_splits', None, "Few-shot splits")


def get_splits(split):
  if FLAGS.fsl_splits is not None:
    train, valid, test = FLAGS.fsl_splits.split('/', maxsplit=3)
    split_patterns = {
      "train": f"test{train}",
      "validation": f"test{valid}",
      "test": f"test{test}",
    }
  else:
    split_patterns = {s: s for s in ("train", "validation", "test")}
  return split_patterns[split]


@datasets.register("mediqa")
class MediqaDataset(PublicSupervisedTFDSDataset):
  """MEDIQA extractive summarization dataset."""

  def load(self, build, split, shuffle):
    split = get_splits(split)
    dataset, info = tfds.load(
      self.override_build(build),
      as_supervised=False,
      split=split,
      with_info=True,
      shuffle_files=shuffle,
      data_dir=self.data_dir)
    return dataset, info.splits[split].num_examples

  def override_build(self, build):
    return "chiqa" + build.lstrip("mediqa")

  def transform(self, dataset):
    return dataset.map(
      lambda d: {
        "inputs": tf.strings.join(
          [
            d['question'],
            'summarize:',
            tf.strings.reduce_join(d["article"], axis=-1, separator=" ")
          ],
          separator=" "),
        "targets": d["summary"],
        "supervised": tf.constant(self.is_supervised)
      })


@datasets.register("bioasq")
class BioAsqDataset(PublicSupervisedTFDSDataset):
  """BioASQ summarization dataset."""

  def load(self, build, split, shuffle):
    split_patterns = {
      "train": "train[:90%]",
      "validation": "train[90%:]",
      "test": "test"
    }
    split = split_patterns[split]
    dataset, info = tfds.load(
      self.override_build(build),
      as_supervised=False,
      split=split,
      with_info=True,
      shuffle_files=shuffle,
      data_dir=self.data_dir)
    return dataset, info.splits[split].num_examples

  def transform(self, dataset):
    return dataset.map(
      lambda d: {
        "inputs": tf.strings.join(
          [
            d['question'],
            'summarize:',
            tf.strings.reduce_join(d["article"], axis=-1, separator=" ")
          ],
          separator=" "),
        "targets": d["summary"],
        "supervised": tf.constant(self.is_supervised)
      })


@datasets.register("duc")
class DucDataset(PublicSupervisedTFDSDataset):
  """DUC summarization dataset."""

  def load(self, build, split, shuffle):
    split = get_splits(split)
    dataset, info = tfds.load(
      self.override_build(build),
      as_supervised=False,
      split=split,
      with_info=True,
      shuffle_files=shuffle,
      data_dir=self.data_dir)
    return dataset, info.splits[split].num_examples

  def transform(self, dataset):
    return dataset.map(
      lambda d: {
        "inputs": tf.strings.join(
          [
            d['question'],
            'summarize:',
            tf.strings.reduce_join(d["document"], axis=-1, separator=" ")
          ],
          separator=" "),
        "targets": d["summary"],
        "supervised": tf.constant(self.is_supervised)
      })


@datasets.register("tac")
class TacDataset(PublicSupervisedTFDSDataset):
  """TAC summarization dataset."""

  def load(self, build, split, shuffle):
    split = get_splits(split)
    dataset, info = tfds.load(
      self.override_build(build),
      as_supervised=False,
      split=split,
      with_info=True,
      shuffle_files=shuffle,
      data_dir=self.data_dir)
    return dataset, info.splits[split].num_examples

  def transform(self, dataset):
    return dataset.map(
      lambda d: {
        "inputs": tf.strings.join(
          [
            d['topic'],
            'summarize:',
            tf.strings.reduce_join(d["articles"], axis=-1, separator=" ")
          ],
          separator=" "),
        "targets": d["summary"],
        "supervised": tf.constant(self.is_supervised)
      })
