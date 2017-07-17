#!/bin/bash

source scripts/bigdl.sh

/home/ubuntu/drizzle-spark/bin/spark-submit \
--verbose \
--master spark://ip-10-0-0-98.us-west-2.compute.internal:7077 \
--driver-memory 20g \
--executor-memory 20g \
--driver-class-path /home/ubuntu/BigDL/spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.vgg.Train \
/home/ubuntu/BigDL/spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
-f /mnt/ebs-vol/cifar-10-batches/cifar-10-batches-bin \
--batchSize 896 \
--partitionNum 32 \
--nodeNum 32 \
--corePerTask 4 \
--checkpoint /mnt/ebs-vol/vgg-model \
--maxEpoch 2
