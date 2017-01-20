Construct a GBT classifier and stuff

REFERENCE:

[1] https://github.com/apache/spark/blob/v1.6.0/sql/core/src/main/scala/org/apache/spark/sql/DataFrameNaFunctions.scala

[2] Mixin Class Composition: http://www.scala-lang.org/old/node/117.html

[3] Setup a standalone cluster
    1) Type start-master.sh
    2) In a webbrowser, go to localhost:8080 and copy the <SPARK_URL> for this session.
    3) Type start-slave.sh <SPARK_URL>
    NOTE: under conf/spark-env.sh, add SPARK_MASTER_IP='<your-ip>' 
