# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Script to evaluate a trained Attention OCR model.

A simple usage example:
python eval.py
"""
import tensorflow as tf
import tf_slim as slim
from tensorflow import app
from tensorflow.python.platform import flags

import data_provider
import common_flags

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('num_batches', 100,
                     'Number of batches to run eval for.')

flags.DEFINE_string('eval_log_dir', '/tmp/attention_ocr/eval',
                    'Directory where the evaluation results are saved to.')

flags.DEFINE_string('checkpoint_path', '',
                    'Path to checkpoint.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency in seconds to run evaluations.')

flags.DEFINE_integer('number_of_steps', None,
                     'Number of times to run evaluation.')
# yapf: enable


def main(_):
  if not tf.io.gfile.exists(FLAGS.eval_log_dir):
    tf.io.gfile.makedirs(FLAGS.eval_log_dir)

  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(dataset.num_char_classes,
                                    dataset.max_sequence_length,
                                    dataset.num_of_views, dataset.null_code)
  data = data_provider.get_data(
      dataset,
      FLAGS.batch_size,
      augment=False,
      central_crop_size=common_flags.get_crop_size())
  endpoints = model.create_base(data.images, labels_one_hot=None)
  model.create_loss(data, endpoints)
  eval_ops = model.create_summaries(
      data, endpoints, dataset.charset, is_training=False)
  slim.get_or_create_global_step()
  session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})

  metric_values = slim.evaluate_once(
      master=FLAGS.master,
      checkpoint_path=FLAGS.checkpoint_path,
      logdir=FLAGS.eval_log_dir,
      num_evals=FLAGS.num_batches,
      initial_op=initial_op,
      eval_op=eval_ops,
      #final_op=name_to_values.values()
      session_config=session_config)

  #for metric, value in zip(names_to_values.keys(), metric_values):
  #  logging.info('Metric %s has value: %f', metric, value)

if __name__ == '__main__':
  app.run()
