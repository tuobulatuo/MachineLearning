package project;

import algorithms.parameterestimate.MixtureGaussianEM;
import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import model.supervised.bagging.BaggingClassification;
import model.supervised.cart.ClassificationTree;
import model.supervised.generative.MixtureGaussianDiscriminantAnalysis;
import model.supervised.neuralnetwork.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import utils.array.ArraySumUtil;
import utils.random.RandomUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/12/15 for machine_learning.
 */
public class Ensemble {

    private static Logger log = LogManager.getLogger(Ensemble.class);

    public static void ensemble1(String path) throws Exception{

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

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int[] structure = {149, 20, 39};
        boolean biased = true;
        NeuralNetwork.MAX_THREADS = 7;
        NeuralNetwork.THREAD_WORK_LOAD = 500;
        NeuralNetwork.BUCKET_COUNT = 220;
        NeuralNetwork.ALPHA = 0.0275;
//        NeuralNetwork.LAMBDA = 0.0001;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 4000;
        NeuralNetwork.PRINT_GAP = 4000;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.0001;

        int trainSize = 878049;
        int testSize = 884262;
        int partition = 2;


        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet1 = dataset.subDataSetByRow(kFoldIndex[0]);

        NeuralNetwork nn = new NeuralNetwork(structure, biased);
        nn.initialize(miniTrainSet1);
        nn.train();

        ClassificationTree.MAX_DEPTH = 8;
        ClassificationTree.MAX_THREADS = 1;
        ClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        ClassificationTree.THREAD_WORK_LOAD = Integer.MAX_VALUE;
        BaggingClassification.MAX_THREADS = 5;
        BaggingClassification.SAMPLE_SIZE_COEF = 0.02;

        DataSet miniTrainSet2 = dataset.subDataSetByRow(kFoldIndex[1]);
        BaggingClassification baggingClassification = new BaggingClassification();
        baggingClassification.initialize(miniTrainSet2);
        String baggingClassName = "model.supervised.cart.ClassificationTree";
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        baggingClassification.baggingConfig(50, baggingClassName, evaluator, trainSet);
        baggingClassification.train();

//        MixtureGaussianEM.MAX_ROUND = 50;
//        MixtureGaussianEM.PRINT_GAP = 50;
//        MixtureGaussianEM.THRESHOLD = 0.000001;
//        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 5;
//        MixtureGaussianDiscriminantAnalysis.MAX_THREADS = 4;
//
//        MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();
//        mixGDA.initialize(miniTrainSet1);
//        mixGDA.train();

        double loss = 0;
        for (int j = 0; j < trainSet.getInstanceLength(); j++) {
            double y = trainSet.getLabel(j);
            double[] yVector = new double[structure[structure.length - 1]];
            yVector[(int) y] = 1;

            double[] feature = trainSet.getInstance(j);
            double[] probs1 = nn.probs(feature);
            double[] probs2 = baggingClassification.probs(feature);
//            double[] probs3 = mixGDA.probs(feature);
            double[] probs = new double[probs1.length];
            IntStream.range(0, probs.length).forEach(k -> probs[k] = probs1[k] + probs2[k]);

            ArraySumUtil.normalize(probs);
            for (int k = 0; k < yVector.length; k++) {
                loss += - yVector[k] * Math.log(probs[k]);
            }

        }

        log.info("train avg loss {}", loss / (double) trainSet.getInstanceLength());

        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());
        String probsPredictsPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.probs.predicts.txt";
        BufferedWriter writer = new BufferedWriter(new FileWriter(probsPredictsPath), 1024 * 1024 * 32);

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

            double[] feature = testSet.getInstance(i);
            double[] probs1 = nn.probs(feature);
            double[] probs2 = baggingClassification.probs(feature);
            double[] probs = new double[probs1.length];
            IntStream.range(0, probs.length).forEach(k -> probs[k] = probs1[k] + probs2[k]);

            ArraySumUtil.normalize(probs);
            double[] arrangedProbs = new double[probs.length];
            for (int j = 0; j < probs.length; j++) {
                String className = (String) indexClassMap.get(j);
                int arrangeIndex = idxMap.get(className);
                arrangedProbs[arrangeIndex] = probs[j];
            }

            Arrays.stream(arrangedProbs).forEach(x -> sb.append(x + ","));
            sb.deleteCharAt(sb.length() - 1);
            writer.write(sb.toString() + "\n");
            id ++;
        }
        writer.close();
    }

//    public static void ensemble2(String path) throws Exception{
//
//        String sep = "\t";
//        boolean hasHeader = false;
//        boolean needBias = true;
//        int m = 51;
//        int n = 878049;
//        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7};
//        boolean isClassification = true;
//
//        Builder builder =
//                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);
//
//        builder.build();
//
//        DataSet dataset = builder.getDataSet();
//        dataset.meanVarianceNorm();
//
//        int[] structure = {137, 30, 39};
//        boolean biased = true;
//        NeuralNetwork.MAX_THREADS = 7;
//        NeuralNetwork.THREAD_WORK_LOAD = 500;
//        NeuralNetwork.BUCKET_COUNT = 220;
//        NeuralNetwork.ALPHA = 0.025;
////        NeuralNetwork.LAMBDA = 0.0001;
//        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
//        NeuralNetwork.MAX_ROUND = 4000;
//        NeuralNetwork.PRINT_GAP = 1000;
//        NeuralNetwork.PRINT_HIDDEN = false;
//        NeuralNetwork.EPSILON = 0.0001;
//
//        int trainSize = 878049;
//        int testSize = 884262;
//        int partition = 3;
//
//
//        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
//
//        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
//        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);
//
//        NeuralNetwork nn = new NeuralNetwork(structure, biased);
//        nn.initialize(miniTrainSet);
//        nn.train();
//
//        ClassificationTree.MAX_DEPTH = 7;
//        ClassificationTree.MAX_THREADS = 1;
//        ClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
//        ClassificationTree.THREAD_WORK_LOAD = Integer.MAX_VALUE;
//        BaggingClassification.MAX_THREADS = 5;
//        BaggingClassification.SAMPLE_SIZE_COEF = 0.02;
//
//        DataSet miniTrainSet2 = dataset.subDataSetByRow(kFoldIndex[1]);
//        BaggingClassification baggingClassification = new BaggingClassification();
//        baggingClassification.initialize(miniTrainSet2);
//        String baggingClassName = "model.supervised.cart.ClassificationTree";
//        ClassificationEvaluator evaluator = new ClassificationEvaluator();
//        baggingClassification.baggingConfig(50, baggingClassName, evaluator, trainSet);
//        baggingClassification.train();
//
////        MixtureGaussianEM.MAX_ROUND = 50;
////        MixtureGaussianEM.PRINT_GAP = 50;
////        MixtureGaussianEM.THRESHOLD = 0.001;
////        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 2;
////        MixtureGaussianDiscriminantAnalysis.MAX_THREADS = 4;
////
////        DataSet miniTrainSet3 = dataset.subDataSetByRow(kFoldIndex[2]);
////        MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();
////        mixGDA.initialize(miniTrainSet3);
////        mixGDA.train();
//
//        int loss = 0;
//        for (int j = 0; j < trainSet.getInstanceLength(); j++) {
//            double y = trainSet.getLabel(j);
//            double[] yVector = new double[structure[structure.length - 1]];
//            yVector[(int) y] = 1;
//
//            double[] feature = trainSet.getInstance(j);
//            double[] probs1 = nn.probs(feature);
//            double[] probs2 = baggingClassification.probs(feature);
////            double[] probs3 = mixGDA.probs(feature);
//            double[] probs = new double[probs1.length];
//            IntStream.range(0, probs.length).forEach(k -> probs[k] = probs1[k] + probs2[k]);
//
//            ArraySumUtil.normalize(probs);
//            for (int k = 0; k < yVector.length; k++) {
//                loss += - yVector[k] * Math.log(probs[k]);
//            }
//
//        }
//
//        log.info("train avg loss {}", loss / (double) trainSet.getInstanceLength());
//
//
//        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());
//        String probsPredictsPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.probs.predicts.txt";
//        BufferedWriter writer = new BufferedWriter(new FileWriter(probsPredictsPath), 1024 * 1024 * 32);
//
//        String head = "Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS";
//        writer.write(head+"\n");
//
//        Map<String, Integer> idxMap = new HashMap<>();
//        String[] es = head.split(",");
//        for (int i = 1; i < es.length; i++) {
//            idxMap.put(es[i], i - 1);
//        }
//
//        Map<Integer, Object> indexClassMap = dataset.getLabels().getIndexClassMap();
//
//        int id = 0;
//        for (int i = 0; i < testSet.getInstanceLength(); i++) {
//
//            StringBuilder sb = new StringBuilder(id + ",");
//
//            double[] feature = testSet.getInstance(i);
//            double[] probs1 = nn.probs(feature);
//            double[] probs2 = baggingClassification.probs(feature);
//            double[] probs = new double[probs1.length];
//            IntStream.range(0, probs.length).forEach(k -> probs[k] = probs1[k] + probs2[k]);
//
//            ArraySumUtil.normalize(probs);
//            double[] arrangedProbs = new double[probs.length];
//            for (int j = 0; j < probs.length; j++) {
//                String className = (String) indexClassMap.get(j);
//                int arrangeIndex = idxMap.get(className);
//                arrangedProbs[arrangeIndex] = probs[j];
//            }
//
//            Arrays.stream(arrangedProbs).forEach(x -> sb.append(x + ","));
//            sb.deleteCharAt(sb.length() - 1);
//            writer.write(sb.toString() + "\n");
//            id ++;
//        }
//        writer.close();
//    }

    public static void main(String[] args) throws Exception{

//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/laplace/data.all.expand.txt";
        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/posterior.5/data.all.expand.txt";
        ensemble1(path2);
//        ensemble2(path);
    }


}
