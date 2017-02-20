workers = ["altoum:2222",
           "edmondo:2222",
           "moschina:2222",
           "dancairo:2222",
           "remendado:2222",]
parameter_servers = ["doncurzio:2222"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers,
                                "worker":workers})
