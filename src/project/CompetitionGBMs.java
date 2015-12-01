package project;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.gradiantboost.GradientBoostClassificationV2;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
import model.supervised.neuralnetwork.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import utils.random.RandomUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/8/15 for machine_learning.
 */
public class CompetitionGBMs {

    private static Logger log = LogManager.getLogger(CompetitionGBMs.class);

    public static void GBMsTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 52;
        int n = 1762311;
        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7,8};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();
        System.gc();

        DataSet dataset = builder.getDataSet();

        int trainSize = 878049;
        int testSize = 884262;
        int partition = 50;


        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);

        GradientBoostClassificationV2.MAX_THREADS = 4;
        GradientBoostClassificationV2.LEARNING_RATE = 1;
        GradientRegressionTree.MAX_THREADS = 1;
        GradientRegressionTree.MAX_DEPTH = 1;
        int boostRound = 10;

        GradientBoostClassificationV2 boostClassification = new GradientBoostClassificationV2();
        boostClassification.initialize(miniTrainSet);
        String boostClassName = "model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree";
        boostClassification.boostConfig(boostRound, boostClassName, new ClassificationEvaluator(), null);
        boostClassification.train();

        ClassificationEvaluator.THREAD_WORK_LOAD = 50000;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(miniTrainSet, boostClassification);
        evaluator.getPredictLabelByProbs();
        log.info("miniTrainSet accu: {}", evaluator.evaluate());
        log.info("miniTrainSet log loss {}", evaluator.logLoss());


        TIntHashSet validateIndex = new TIntHashSet(trainSize);
        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, boostClassification);
        evaluator.getPredictLabelByProbs();
        log.info("validate accu: {}", evaluator.evaluate());
        log.info("validate log loss: {}", evaluator.logLoss());


        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());

        evaluator.initialize(testSet, boostClassification);
        evaluator.getPredictLabelByProbs();
        double[][] probs = evaluator.getProbs();

        String probsPredictsPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.probs.predicts.txt";
        BufferedWriter writer = new BufferedWriter(new FileWriter(probsPredictsPath), 1024 * 1024 * 64);

        String head = "Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS";
        writer.write(head+"\n");

        Map<String, Integer> idxMap = new HashMap<>();
        String[] es = head.split(",");
        for (int i = 1; i < es.length; i++) {
            idxMap.put(es[i], i - 1);
        }

        Map<Integer, Object> indexClassMap = dataset.getLabels().getIndexClassMap();

        int id = 0;
        for (int i = 0; i < testSet.getInstanceLength(); i++) {

            StringBuilder sb = new StringBuilder(id + ",");

            double[] probsI = probs[i];
            double[] arrangedProbs = new double[probsI.length];
            for (int j = 0; j < probsI.length; j++) {
                String className = (String) indexClassMap.get(j);
                int arrangeIndex = idxMap.get(className);
                arrangedProbs[arrangeIndex] = probsI[j];
            }

            Arrays.stream(arrangedProbs).forEach(x -> sb.append(x + ","));
            sb.deleteCharAt(sb.length() - 1);
            writer.write(sb.toString() + "\n");

            if (id ++ % 100000 == 0) log.info("write {} lines ..", id);
        }
        writer.close();

        log.info("task done, exit ..");
    }



    public static void main(String[] args) throws Exception{

//        String prior = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/prior/data.all.expand.txt";
//        neuralNetworkTest(prior);

        String nobuzz = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/nobuzz/data.all.expand.txt";
        GBMsTest(nobuzz);
    }
}
