/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.lenet

import java.nio.file.Paths

import scala.util.Random

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, _}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import com.intel.analytics.bigdl.utils.RandomGenerator._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Train {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {

    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf()
        .setAppName("Train Lenet on MNIST")
        .set("spark.task.maxFailures", "1")
        .set("spark.executor.drizzle.barrierAcrossBatches", "true")
      val sc = new SparkContext(conf)
      Engine.init(param.nodeNumber, param.coreNumber, true)

      Engine.setPartitionNumber(Some(param.partitionNum))
      Engine.setCoreNumber(param.coreNumber)

      val trainData = Paths.get(param.folder, "/train-images-idx3-ubyte")
      val trainLabel = Paths.get(param.folder, "/train-labels-idx1-ubyte")
      val validationData = Paths.get(param.folder, "/t10k-images-idx3-ubyte")
      val validationLabel = Paths.get(param.folder, "/t10k-labels-idx1-ubyte")

      if (param.seed.isDefined) {
         Random.setSeed(param.seed.get)
         RNG.setSeed(param.seed.get)
      }

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        LeNet5(classNum = 10)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> param.learningRate,
          "learningRateDecay" -> param.learningRateDecay,
          "drizzleGroupSize" -> param.drizzleGroupSize,
          "useDrizzle" -> param.useDrizzle
        )
      }

      val trainSet = DataSet.array(load(trainData, trainLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(
        param.batchSize)

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = ClassNLLCriterion[Float]())
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
        param.batchSize)

      optimizer
        .setValidation(
          trigger = Trigger.everyEpoch,
          dataset = validationSet,
          vMethods = Array(new Top1Accuracy, new Top5Accuracy[Float], new Loss[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
