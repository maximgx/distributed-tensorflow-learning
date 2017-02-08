  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)
