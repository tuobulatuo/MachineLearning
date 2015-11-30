package project;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.gradiantboost.GradientBoostClassification;
import model.supervised.boosting.gradiantboost.GradientBoostRegression;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
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
public class ModelTests {

    private static Logger log = LogManager.getLogger(ModelTests.class);

    public static void neuralNetworkTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 50;
        int n = 878049;
        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7,8};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int[] structure = {137, 20, 39};
        boolean biased = true;
        NeuralNetwork.MAX_THREADS = 7;
        NeuralNetwork.THREAD_WORK_LOAD = 500;
        NeuralNetwork.BUCKET_COUNT = 220;
        NeuralNetwork.ALPHA = 0.0275 / (double) NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.LAMBDA = 0.0001 / (double) NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 3800;
        NeuralNetwork.PRINT_GAP = 3800;
        NeuralNetwork.PRINT_HIDDEN = false;
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

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(miniTrainSet, nn);
        evaluator.getPredictLabelByProbs();
        log.info("miniTrainSet accu: {}", evaluator.evaluate());

        double accu = 0;
        for (int j = 0; j < miniTrainSet.getInstanceLength(); j++) {
            double y = miniTrainSet.getLabel(j);
            double[] yVector = new double[structure[structure.length - 1]];
            yVector[(int) y] = 1;

            double[] feature = miniTrainSet.getInstance(j);
            double[] probs = nn.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("miniTrainSet avg loss {}", accu / (double) miniTrainSet.getInstanceLength());


        TIntHashSet validateIndex = new TIntHashSet(trainSize);
        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, nn);
        evaluator.getPredictLabelByProbs();
        log.info("validate accu: {}", evaluator.evaluate());

        accu = 0;
        for (int j = 0; j < validateSet.getInstanceLength(); j++) {
            double y = validateSet.getLabel(j);
            double[] yVector = new double[structure[structure.length - 1]];
            yVector[(int) y] = 1;

            double[] feature = validateSet.getInstance(j);
            double[] probs = nn.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("validate avg loss {}", accu / (double) validateSet.getInstanceLength());


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

            double[] probs = nn.probs(testSet.getInstance(i));
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

    public static void pcaNeuralNetworkTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 27;
        int n = 877982;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int[] structure = {28, 20, 39};
        boolean biased = true;
        NeuralNetwork.MAX_THREADS = 7;
        NeuralNetwork.THREAD_WORK_LOAD = 250;
        NeuralNetwork.BUCKET_COUNT = 440;
        NeuralNetwork.ALPHA = 0.06 / (double) NeuralNetwork.BUCKET_COUNT;

        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 40000;
        NeuralNetwork.PRINT_GAP = 5000;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.0001;

        int trainSize = 877982;
        int testSize = 884262;
        int partition = 2;


        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);

        NeuralNetwork nn = new NeuralNetwork(structure, biased);
        nn.initialize(miniTrainSet);
        nn.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(miniTrainSet, nn);
        evaluator.getPredictLabelByProbs();
        log.info("miniTrainSet accu: {}", evaluator.evaluate());

        double accu = 0;
        for (int j = 0; j < miniTrainSet.getInstanceLength(); j++) {
            double y = miniTrainSet.getLabel(j);
            double[] yVector = new double[structure[structure.length - 1]];
            yVector[(int) y] = 1;

            double[] feature = miniTrainSet.getInstance(j);
            double[] probs = nn.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("miniTrainSet avg loss {}", accu / (double) miniTrainSet.getInstanceLength());


        TIntHashSet validateIndex = new TIntHashSet(877982);
        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, nn);
        evaluator.getPredictLabelByProbs();
        log.info("validate accu: {}", evaluator.evaluate());

        accu = 0;
        for (int j = 0; j < validateSet.getInstanceLength(); j++) {
            double y = validateSet.getLabel(j);
            double[] yVector = new double[structure[structure.length - 1]];
            yVector[(int) y] = 1;

            double[] feature = validateSet.getInstance(j);
            double[] probs = nn.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("validate avg loss {}", accu / (double) validateSet.getInstanceLength());


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

            double[] probs = nn.probs(testSet.getInstance(i));
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

    public static void GradientBoostTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 60;
        int n = 877982;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        GradientBoostClassification.NEED_REPORT = true;
        GradientBoostClassification.MAX_THREADS = 4;
        GradientRegressionTree.MAX_THREADS = 1;
        GradientRegressionTree.MAX_DEPTH = 4;
        GradientBoostRegression.OUTLIER_QUANTILE = 100;
        int boostRound = 3;

        int trainSize = 877982;
        int testSize = 884262;
        int partition = 200;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);


        GradientBoostClassification boostClassification = new GradientBoostClassification();
        boostClassification.initialize(miniTrainSet);
        String boosterClassName = "model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree";
        boostClassification.boostConfig(boostRound, boosterClassName, new Evaluator(), miniTrainSet);
        boostClassification.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(miniTrainSet, boostClassification);
        evaluator.getPredictLabelByProbs();
        log.info("miniTrainSet accu: {}", evaluator.evaluate());

        double accu = 0;
        for (int j = 0; j < miniTrainSet.getInstanceLength(); j++) {
            double y = miniTrainSet.getLabel(j);
            double[] yVector = new double[miniTrainSet.getLabels().getClassIndexMap().size()];
            yVector[(int) y] = 1;

            double[] feature = miniTrainSet.getInstance(j);
            double[] probs = boostClassification.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("miniTrainSet avg loss {}", accu / (double) miniTrainSet.getInstanceLength());

        TIntHashSet validateIndex = new TIntHashSet(877982);
        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, boostClassification);
        evaluator.getPredictLabelByProbs();
        log.info("validate accu: {}", evaluator.evaluate());

        accu = 0;
        for (int j = 0; j < validateSet.getInstanceLength(); j++) {
            double y = validateSet.getLabel(j);
            double[] yVector = new double[trainSet.getLabels().getClassIndexMap().size()];
            yVector[(int) y] = 1;

            double[] feature = validateSet.getInstance(j);
            double[] probs = boostClassification.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("validate avg loss {}", accu / (double) validateSet.getInstanceLength());


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
//            double[] probs = boostClassification.probs(testSet.getInstance(i));
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
    }

    public static void main(String[] args) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.all.txt";

        neuralNetworkTest(path);

//        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.all.pca.50.txt";

//        pcaNeuralNetworkTest(path2);

//        GradientBoostTest(path2);

//        mixtureGDATest(path);
    }

//    public static void mixtureGDATest(String path) throws Exception{
//
//        String sep = "\t";
//        boolean hasHeader = false;
//        boolean needBias = false;
//        int m = 50;
//        int n = 877982;
//        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7};
//        boolean isClassification = true;
//
//        Builder builder =
//                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);
//
//        builder.build();
//
//        DataSet dataset = builder.getDataSet();
//
//        MixtureGaussianEM.MAX_ROUND = 50;
//        MixtureGaussianEM.PRINT_GAP = 50;
//        MixtureGaussianEM.THRESHOLD = 0.001;
//        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 2;
//        MixtureGaussianDiscriminantAnalysis.MAX_THREADS = 4;
//
//        int trainSize = 878049;
//        int testSize = 884262;
//        int partition = 2;
//
//        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
//
//        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
//        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);
//
//        MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();
//        mixGDA.initialize(trainSet);
//        mixGDA.train();
//
//        ClassificationEvaluator evaluator = new ClassificationEvaluator();
////        evaluator.initialize(miniTrainSet, mixGDA);
////        evaluator.getPredictLabelByProbs();
////        log.info("miniTrainSet accu: {}", evaluator.evaluate());
////
//        double accu = 0;
////        for (int j = 0; j < miniTrainSet.getInstanceLength(); j++) {
////            double y = miniTrainSet.getLabel(j);
////            double[] yVector = new double[trainSet.getLabels().getClassIndexMap().size()];
////            yVector[(int) y] = 1;
////
////            double[] feature = miniTrainSet.getInstance(j);
////            double[] probs = mixGDA.probs(feature);
////
////            for (int k = 0; k < yVector.length; k++) {
////                accu += - yVector[k] * Math.log(probs[k]);
////            }
////        }
////
////        log.info("miniTrainSet avg loss {}", accu / (double) miniTrainSet.getInstanceLength());
//
//
//        TIntHashSet validateIndex = new TIntHashSet(trainSize);
//        IntStream.range(1, kFoldIndex.length).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
//        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
//        evaluator.initialize(validateSet, mixGDA);
//        evaluator.getPredictLabelByProbs();
//        log.info("validate accu: {}", evaluator.evaluate());
//
//        accu = 0;
//        for (int j = 0; j < validateSet.getInstanceLength(); j++) {
//            double y = validateSet.getLabel(j);
//            double[] yVector = new double[trainSet.getLabels().getClassIndexMap().size()];
//            yVector[(int) y] = 1;
//
//            double[] feature = validateSet.getInstance(j);
//            double[] probs = mixGDA.probs(feature);
//
//            for (int k = 0; k < yVector.length; k++) {
//                accu += - yVector[k] * Math.log(probs[k]);
//            }
//        }
//
//        log.info("validate avg loss {}", accu / (double) validateSet.getInstanceLength());
//
//
////        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());
////        String probsPredictsPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.probs.predicts.txt";
////        BufferedWriter writer = new BufferedWriter(new FileWriter(probsPredictsPath), 1024 * 1024 * 32);
////
////        String head = "Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS";
////        writer.write(head+"\n");
////
////        Map<String, Integer> idxMap = new HashMap<>();
////        String[] es = head.split(",");
////        for (int i = 1; i < es.length; i++) {
////            idxMap.put(es[i], i - 1);
////        }
////
////        Map<Integer, Object> indexClassMap = dataset.getLabels().getIndexClassMap();
////
////        int id = 0;
////        for (int i = 0; i < testSet.getInstanceLength(); i++) {
////
////            StringBuilder sb = new StringBuilder(id + ",");
////
////            double[] probs = mixGDA.probs(testSet.getInstance(i));
////            double[] arrangedProbs = new double[probs.length];
////            for (int j = 0; j < probs.length; j++) {
////                String className = (String) indexClassMap.get(j);
////                int arrangeIndex = idxMap.get(className);
////                arrangedProbs[arrangeIndex] = probs[j];
////            }
////
////            Arrays.stream(arrangedProbs).forEach(x -> sb.append(x + ","));
////            sb.deleteCharAt(sb.length() - 1);
////            writer.write(sb.toString() + "\n");
////            id ++;
////        }
////        writer.close();
//    }



    //    public static void bagging(String path) throws Exception{
//
//        String sep = "\t";
//        boolean hasHeader = false;
//        boolean needBias = true;
//        int m = 51;
//        int n = 877982;
//        int[] featureCategoryIndex = {0, 1, 2, 3, 4, 5, 6, 7};
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
//        int trainSize = 877982;
//
//
//        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
//        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, 100);
//        DataSet miniSet = trainSet.subDataSetByRow(kFoldIndex[0]);
//
//        ClassificationTree.MAX_DEPTH = 7;
//        ClassificationTree.MAX_THREADS = 1;
//        ClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
//        ClassificationTree.THREAD_WORK_LOAD = Integer.MAX_VALUE;
//        BaggingClassification.MAX_THREADS = 5;
//        BaggingClassification.SAMPLE_SIZE_COEF = 1;
//
//        BaggingClassification baggingClassification = new BaggingClassification();
//        baggingClassification.initialize(miniSet);
//        String baggingClassName = "model.supervised.cart.ClassificationTree";
//        Evaluator evaluator = new ClassificationEvaluator();
//        baggingClassification.baggingConfig(50, baggingClassName, evaluator, miniSet);
//        baggingClassification.train();
//
//
//        evaluator.initialize(trainSet, baggingClassification);
//        evaluator.getPredictLabelByProbs();
//        log.info("train accu: {}", evaluator.evaluate());
//
//        double accu = 0;
//        for (int j = 0; j < trainSet.getInstanceLength(); j++) {
//            double y = trainSet.getLabel(j);
//            double[] yVector = new double[trainSet.getLabels().getClassIndexMap().size()];
//            yVector[(int) y] = 1;
//
//            double[] feature = trainSet.getInstance(j);
//            double[] probs = baggingClassification.probs(feature);
//
//            for (int k = 0; k < yVector.length; k++) {
//                accu += - yVector[k] * Math.log(probs[k]);
//            }
//        }
//
//        log.info("trainSet avg loss {}", accu / (double) trainSet.getInstanceLength());
//
//        int testSize = 884262;
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
//            double[] probs = baggingClassification.probs(testSet.getInstance(i));
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
}
