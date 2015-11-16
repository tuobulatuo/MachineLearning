package project;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.bagging.BaggingClassification;
import model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump;
import model.supervised.boosting.gradiantboost.GradientBoostClassification;
import model.supervised.boosting.gradiantboost.GradientBoostRegression;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
import model.supervised.cart.ClassificationTree;
import model.supervised.ecoc.ECOCAdaBoost;
import model.supervised.naivebayes.Multinoulli;
import model.supervised.neuralnetwork.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;
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
public class CompetitionLaplace {

    private static Logger log = LogManager.getLogger(CompetitionLaplace.class);

    public static void neuralNetworkTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 51;
        int n = 878049;
        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7,8};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();
        System.gc();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int[] structure = {149, 30, 39};
        boolean biased = true;
        NeuralNetwork.MAX_THREADS = 4;
        NeuralNetwork.THREAD_WORK_LOAD = 500;
        NeuralNetwork.BUCKET_COUNT = 220;
        NeuralNetwork.ALPHA = 0.25;
//        NeuralNetwork.LAMBDA = 0.001;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 10000;
        NeuralNetwork.PRINT_GAP = 10000;
        NeuralNetwork.EPSILON = 0.0001;

        int trainSize = 878049;
        int testSize = 884262;
        int partition = 2;


        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);

        NeuralNetwork nn = new NeuralNetwork(structure, biased);
        nn.initialize(miniTrainSet);
        nn.train();

        ClassificationEvaluator.THREAD_WORK_LOAD = 50000;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(miniTrainSet, nn);
        evaluator.getPredictLabel();
        log.info("miniTrainSet accu: {}", evaluator.evaluate());
        log.info("miniTrainSet log loss {}", evaluator.logLoss());

//        double loss = 0;
//        for (int i = 0; i < miniTrainSet.getInstanceLength(); i++) {
//            double y = miniTrainSet.getLabel(i);
//            double[] feature = miniTrainSet.getInstance(i);
//            double[] probs = nn.probs(feature);
//            loss -= Math.log(probs[(int) y]);
//        }
//
//        log.info("miniTrainSet avg loss {}", loss / (double) miniTrainSet.getInstanceLength());


        TIntHashSet validateIndex = new TIntHashSet(trainSize);
        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, nn);
        evaluator.getPredictLabel();
        log.info("validate accu: {}", evaluator.evaluate());
        log.info("validate log loss: {}", evaluator.logLoss());

//        accu = 0;
//        for (int j = 0; j < validateSet.getInstanceLength(); j++) {
//            double y = validateSet.getLabel(j);
//            double[] yVector = new double[structure[structure.length - 1]];
//            yVector[(int) y] = 1;
//
//            double[] feature = validateSet.getInstance(j);
//            double[] probs = nn.probs(feature);
//
//            for (int k = 0; k < yVector.length; k++) {
//                accu += - yVector[k] * Math.log(probs[k]);
//            }
//        }
//
//        log.info("validate avg loss {}", accu / (double) validateSet.getInstanceLength());


        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());

        evaluator.initialize(testSet, nn);
        evaluator.getPredictLabel();
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
    }



    public static void main(String[] args) throws Exception{

//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/laplace/data.all.expand.txt";
//        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/posterior.5/data.all.expand.txt";

        String path3 = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/nobuzz/data.all.expand.txt";

        neuralNetworkTest(path3);

//        ecocTest(path3);

    }

    public static void ecocTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 51;
        int n = 878049;
        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7,8};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();
        System.gc();

        DataSet dataset = builder.getDataSet();

        int trainSize = 878049;
        int testSize = 884262;
        int partition = 200;

        String boostClassName = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
        ECOCAdaBoost.ADABOOST_CLASSIFIER_CLASS_NAME = boostClassName;
        ECOCAdaBoost.MAX_THREADS = 4;
        ECOCAdaBoost.MAX_ITERATION = 20;
        ECOCAdaBoost.DEFAULT_CODE_WORD_LENGTH = 50;

        DecisionStump.MAX_THREADS = 1;
        DecisionStump.THREAD_WORK_LOAD = Integer.MAX_VALUE;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);


        ECOCAdaBoost ecocAdaBoost = new ECOCAdaBoost();
        ecocAdaBoost.initialize(miniTrainSet);
        ecocAdaBoost.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        ClassificationEvaluator.THREAD_WORK_LOAD = 5000;
        evaluator.initialize(miniTrainSet, ecocAdaBoost);
        evaluator.getPredictLabel();

        log.info("miniTrainSet accu: {}", evaluator.evaluate());
        log.info("miniTrainSet avg loss: {}", evaluator.logLoss());

        TIntHashSet validateIndex = new TIntHashSet(trainSize);
        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, ecocAdaBoost);
        evaluator.getPredictLabel();
        log.info("validate accu: {}", evaluator.evaluate());
        log.info("validate avg loss {}", evaluator.logLoss());

//        double accu = 0;
//        int counter = 0;
//        for (int j = 0; j < miniTrainSet.getInstanceLength(); j++) {
//            double y = miniTrainSet.getLabel(j);
//            double[] yVector = new double[trainSet.getLabels().getClassIndexMap().size()];
//            yVector[(int) y] = 1;
//
//            double[] feature = miniTrainSet.getInstance(j);
//            double[] probs = ecocAdaBoost.probs(feature);
//
//            for (int k = 0; k < yVector.length; k++) {
//                accu += - yVector[k] * Math.log(probs[k]);
//            }
//
//            if (counter++ % 10000 == 0) log.info("{} ids ..", counter);
//        }
//
//        log.info("miniTrainSet avg loss {}", accu / (double) miniTrainSet.getInstanceLength());

//        accu = 0;
//        counter = 0;
//        for (int j = 0; j < validateSet.getInstanceLength(); j++) {
//            double y = validateSet.getLabel(j);
//            double[] yVector = new double[trainSet.getLabels().getClassIndexMap().size()];
//            yVector[(int) y] = 1;
//
//            double[] feature = validateSet.getInstance(j);
//            double[] probs = ecocAdaBoost.probs(feature);
//
//            for (int k = 0; k < yVector.length; k++) {
//                accu += - yVector[k] * Math.log(probs[k]);
//            }
//
//            if (counter++ % 10000 == 0) log.info("{} ids ..", counter);
//        }

//        log.info("validate avg loss {}", accu / (double) validateSet.getInstanceLength());

        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());
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

            double[] probs = ecocAdaBoost.probs(testSet.getInstance(i));
            double[] arrangedProbs = new double[probs.length];
            for (int j = 0; j < probs.length; j++) {
                String className = (String) indexClassMap.get(j);
                int arrangeIndex = idxMap.get(className);
                arrangedProbs[arrangeIndex] = probs[j];
            }

            Arrays.stream(arrangedProbs).forEach(x -> sb.append(x + ","));
            sb.deleteCharAt(sb.length() - 1);
            writer.write(sb.toString() + "\n");

            if (id ++ % 100000 == 0) log.info("write {} lines ..", id);
        }
        writer.close();
    }

}
