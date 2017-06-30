#!/bin/bash

if [ $# -ne 2 ];
then
echo "Usage <script> <drizzleGroupSize> <useDrizzle>"
exit 0
fi

drizzleGroupSize=$1
useDrizzle=$2

source scripts/bigdl.sh

/home/ubuntu/drizzle-spark/bin/spark-submit \
--verbose \
--master spark://ec2-34-211-70-169.us-west-2.compute.amazonaws.com:7077 \
--driver-memory 20g \
--executor-memory 20g \
--driver-class-path /home/ubuntu/BigDL/spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.vgg.Train \
/home/ubuntu/BigDL/spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
-f /mnt/ebs-vol/cifar-10-batches/cifar-10-batches-bin \
--batchSize 512 \
--partitionNum 128 \
--nodeNum 32 \
--corePerTask 1 \
--checkpoint /mnt/ebs-vol/vgg-model \
--maxEpoch 2 \
--drizzleGroupSize $drizzleGroupSize \
--useDrizzle $useDrizzle
