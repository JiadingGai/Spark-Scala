#!/bin/bash
mvn clean
mvn package
#spark-submit --class com.vario.demo.Demo --master spark://192.168.1.71:7077 target/demo-0.0.1-SNAPSHOT.jar hdfs://192.168.1.71:50050/user/gaipaul/demo/data/inputfile.txt 2 hdfs://192.168.1.71:50050/user/gaipaul/demo/data/sample_libsvm_data.txt
spark-submit --class com.vario.demo.Demo --master local target/demo-0.0.1-SNAPSHOT.jar data/inputfile.txt 2 data/mllib/sample_libsvm_data.txt
