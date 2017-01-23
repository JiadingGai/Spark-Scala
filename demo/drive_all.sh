#!/bin/bash
mvn clean
mvn package
#spark-shell --master local --packages com.databricks:spark-csv_2.10:1.5.0

#spark-submit --class com.vario.demo.Demo --master spark://192.168.1.71:7077 target/demo-0.0.1-SNAPSHOT.jar hdfs://192.168.1.71:50050/user/gaipaul/demo/data/inputfile.txt 2 hdfs://192.168.1.71:50050/user/gaipaul/demo/data/sample_libsvm_data.txt

spark-submit --class com.vario.demo.Demo --master local --packages com.databricks:spark-csv_2.10:1.5.0 target/demo-0.0.1-SNAPSHOT.jar data/inputfile.txt 2 data/mllib/sample_libsvm_data.txt

#spark-submit --class com.vario.demo.Demo --master spark://127.0.0.1:7077 --packages com.databricks:spark-csv_2.10:1.5.0 target/demo-0.0.1-SNAPSHOT.jar
