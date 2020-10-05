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

"""Binary to perform evaluation of a trained model."""
import csv
import os

from pegasus.eval.text_eval import _BLEU_METRIC, _REPETITION_METRIC, _ROUGE_METRIC, _LENGTH_METRIC, \
  LogWriter, _write_aggregates, _write_aggregate_summaries
from rouge_score import rouge_scorer
from rouge_score import scoring

from pegasus.eval.bleu import bleu_scorer
from pegasus.eval.length import length_scorer
from pegasus.eval.repetition import repetition_scorer

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("predictions", None, "Predictions file")
flags.DEFINE_string("model_dir", None, "Output directory for model checkpoints or the specific model checkpoint.")
flags.DEFINE_boolean("enable_logging", True, "Enable logging of model ouputs.")

def text_eval(preds_file,
              model_dir,
              global_step: int = 0,
              eval_tag: str = "",
              enable_logging: bool = True):
  """Evaluates a set of text targets/predictions."""
  scorers_dict = {
    _ROUGE_METRIC: rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True),
    _BLEU_METRIC: bleu_scorer.BleuScorer(),
    _REPETITION_METRIC: repetition_scorer.RepetitionScorer(["regs1", "regs2", "regs3", "regsTCR"]),
    _LENGTH_METRIC: length_scorer.LengthScorer(["word", "char"])
  }
  aggregators_dict = {k: scoring.BootstrapAggregator() for k in scorers_dict}

  with LogWriter((), model_dir, 0, "", enable_logging) as log_writer:
    with open(preds_file) as csv_file:
      reader = csv.DictReader(csv_file)
      for i, row in enumerate(reader):
        text_dict = {
          "inputs": row['prompt'],
          "targets": row['targets'],
          "predictions": row['predictions']
        }

        log_writer.write(text_dict, i)

        for key, scorer in scorers_dict.items():
          scores_i = scorer.score(row['targets'], row['predictions'])
          aggregators_dict[key].add_scores(scores_i)

  aggregates_dict = {k: v.aggregate() for k, v in aggregators_dict.items()}
  length_histograms = scorers_dict[_LENGTH_METRIC].histograms(as_string=True)
  _write_aggregates(model_dir, global_step, eval_tag, aggregates_dict,
                    length_histograms)
  _write_aggregate_summaries(model_dir, global_step, eval_tag, aggregates_dict)


def main(_):
  model_dir = FLAGS.model_dir or os.path.dirname(FLAGS.predictions)
  text_eval(FLAGS.predictions, model_dir, enable_logging=FLAGS.enable_logging)


if __name__ == "__main__":
  flags.mark_flags_as_required(["predictions"])
  tf.enable_eager_execution()
  tf.app.run(main)
