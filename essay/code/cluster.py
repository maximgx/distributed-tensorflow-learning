parameter_servers = ["pc-01:2222"]
workers = [ "pc-02:2222", 
            "pc-03:2222",
            "pc-04:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers,
				"worker":workers})
