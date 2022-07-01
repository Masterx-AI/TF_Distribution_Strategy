## Overview

The `tf.distribute.Strategy` API provides an abstraction for distributing your training across multiple processing units. It allows you to carry out distributed training using existing models and training code with minimal changes.

This tutorial demonstrates how to use the `tf.distribute.MirroredStrategy` to perform in-graph replication with _synchronous training on many GPUs on one machine_. The strategy essentially copies all of the model's variables to each processor. Then, it uses [all-reduce](http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/) to combine the gradients from all processors, and applies the combined value to all copies of the model.

You will use the `tf.keras` APIs to build the model and `Model.fit` for training it. (To learn about distributed training with a custom training loop and the `MirroredStrategy`, check out [this tutorial](custom_training.ipynb).)

`MirroredStrategy` trains your model on multiple GPUs on a single machine. For _synchronous training on many GPUs on multiple workers_, use the `tf.distribute.MultiWorkerMirroredStrategy` with the [Keras Model.fit](multi_worker_with_keras.ipynb) or [a custom training loop](multi_worker_with_ctl.ipynb). For other options, refer to the [Distributed training guide](../../guide/distributed_training.ipynb).

To learn about various other strategies, there is the [Distributed training with TensorFlow](../../guide/distributed_training.ipynb) guide.
