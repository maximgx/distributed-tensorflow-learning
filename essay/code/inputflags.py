tf.app.flags.DEFINE_string("job_name", "",
			"Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0,
			"Index of task within the job")
FLAGS = tf.app.flags.FLAGS

server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)
