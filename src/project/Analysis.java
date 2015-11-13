package project;

import algorithms.parameterestimate.MixtureGaussianEM;
import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.bagging.BaggingClassification;
import model.supervised.boosting.adaboot.SAMME;
import model.supervised.boosting.adaboot.SAMMESampleSimulation;
import model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree;
import model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump;
import model.supervised.boosting.adaboot.adaboostclassifier.WeightedClassificationTree;
import model.supervised.boosting.gradiantboost.GradientBoostClassification;
import model.supervised.boosting.gradiantboost.GradientBoostRegression;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
import model.supervised.cart.ClassificationTree;
import model.supervised.cart.Tree;
import model.supervised.eoec.EOECAdaBoost;
import model.supervised.generative.MixtureGaussianDiscriminantAnalysis;
import model.supervised.neuralnetwork.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.io.IOException;
import java.util.Map;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class Analysis {

    private static Logger log = LogManager.getLogger(Analysis.class);

    public static void classificationTreeTest(String path) throws IOException {

//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.hour.txt";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 46;
        int n = 87000;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.CONFUSION_MATRIX = false;
        ClassificationEvaluator classifyEva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(classifyEva, dataset, 10, Norm.NULL);

        Tree classificationTree = new ClassificationTree();
        ClassificationTree.MAX_THREADS = 4;
        ClassificationTree.THREAD_WORK_LOAD = 1;
        ClassificationTree.MAX_DEPTH = 30;
        ClassificationTree.INFORMATION_GAIN_THRESHOLD = 0.01;
        ClassificationTree.MIN_INSTANCE_COUNT = 5;
        crossEvaluator.crossValidateEvaluate(classificationTree);

    }

    public static void gradientBoostTest(String path1, int boost) throws Exception{
//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.hour.txt";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 44;
        int n = 87000;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        GradientBoostClassification.NEED_REPORT = true;
        GradientBoostClassification.MAX_THREADS = 4;
        GradientRegressionTree.MAX_THREADS = 1;
        GradientRegressionTree.MAX_DEPTH = 4;
        GradientBoostRegression.OUTLIER_QUANTILE = 100;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            GradientBoostClassification boostClassification = new GradientBoostClassification();
            boostClassification.initialize(trainSet);
            String className = "model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree";
            boostClassification.boostConfig(boost, className, new Evaluator(), trainSet);
            boostClassification.train();


            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(trainSet, boostClassification);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, boostClassification);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

            break;
        }

        // boost = 3 : train error = 0.63
    }

    public static void adaBoostTest(String path) throws Exception{
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 44;
        int n = 11314;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMMESampleSimulation.NEED_ROUND_REPORT = false;
        SAMMESampleSimulation.SAMPLE_SIZE_COEF = 1;
        AdaBoostClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        AdaBoostClassificationTree.MAX_DEPTH = 8;
        AdaBoostClassificationTree.MAX_THREADS = 7;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 100);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            SAMMESampleSimulation sammeSampleSimulation = new SAMMESampleSimulation();
            sammeSampleSimulation.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree";
            sammeSampleSimulation.boostConfig(200, className, new ClassificationEvaluator(), trainSet);
            sammeSampleSimulation.train();

            Evaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(trainSet, sammeSampleSimulation);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, sammeSampleSimulation);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

            break;
        }
        //100 round: train error 0.5, test error 0.8
    }

    public static void adaBoostTest2(String path) throws Exception{
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 44;
        int n = 11314;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = false;
        WeightedClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        WeightedClassificationTree.MAX_DEPTH = 6;
        WeightedClassificationTree.MAX_THREADS = 7;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 100);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.WeightedClassificationTree";
            samme.boostConfig(14, className, new ClassificationEvaluator(), trainSet);
            samme.train();

            Evaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(trainSet, samme);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

            break;
        }
        //100 round: train error 0.5, test error 0.8
    }

    public static void neuralNetworkTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
//        int m = 44;
        int m = 51;
        int n = 877000;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4, 5, 6, 7};
//        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

//        Map classIndexMap = dataset.getLabels().getClassIndexMap();
//        for (Object k : classIndexMap.keySet()){
//            log.info("{} {}", k, classIndexMap.get(k));
//        }
//
//        Map indexClassMap = dataset.getLabels().getIndexClassMap();
//        for (Object k : indexClassMap.keySet()){
//            log.info("{} {}", k, indexClassMap.get(k));
//        }
//        System.exit(0);

        dataset.meanVarianceNorm();

        // 2% + {124, 10, 39} ~ 1289 + ALPHA = 0.005 + MAX_ROUND = 4000  => 0.32 train 0.306 test
        // 5% + {124, 15, 39} ~ 2445  + ALPHA = 0.002 + MAX_ROUND = 4000  => 0.32 train 0.31 test
        // 5% + {124, 20, 39} ~ 2539  + ALPHA = 0.002 + MAX_ROUND = 4000  => 0.32 train 0.311 test
        // 6.6% + {124, 20, 39} ~ 2539  + ALPHA = 0.003 + MAX_ROUND = 4000  => 0.32 train 0.311 test
        // 6.6% + {124, 30, 39} ~ 4890  + ALPHA = 0.003 + MAX_ROUND = 4000  => 0.32 train 0.311 test
        // 20% + {124, 20, 39} ~ 4890  + ALPHA = 0.01 + MAX_ROUND = 10000  => 0.314 train 0.313 test

//        int[] structure = {124, 20, 39};
        int[] structure = {137, 22, 39};
        boolean biased = true;
        NeuralNetwork.MAX_THREADS = 7;
        NeuralNetwork.THREAD_WORK_LOAD = 500;
        NeuralNetwork.BUCKET_COUNT = 220;
        NeuralNetwork.ALPHA = 0.03 / (double) NeuralNetwork.BUCKET_COUNT;
//        NeuralNetwork.ALPHA_SHRINKAGE_RATE = 0.1;
//        NeuralNetwork.ALPHA_SHRINKAGE_ITERATION = 5000;
//        NeuralNetwork.LAMBDA = 0.00001 / NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 13000;
        NeuralNetwork.PRINT_GAP = 1000;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.0001;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 2);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            NeuralNetwork nn = new NeuralNetwork(structure, biased);
            nn.initialize(trainSet);
            nn.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(trainSet, nn);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, nn);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

//            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            double accu = 0;
            for (int j = 0; j < testSet.getInstanceLength(); j++) {
                double y = testSet.getLabel(j);
                double[] yVector = new double[structure[structure.length - 1]];
                yVector[(int) y] = 1;

                double[] feature = testSet.getInstance(j);
                double[] probs = nn.probs(feature);

                for (int k = 0; k < yVector.length; k++) {
                    accu += - yVector[k] * Math.log(probs[k]);
                }
            }

            log.info("test avg loss {}", accu / (double) testSet.getInstanceLength());

            break;
        }
    }

    public static void bagging(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 52;
        int n = 870000;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        ClassificationTree.MAX_DEPTH = 7;
        ClassificationTree.MAX_THREADS = 1;
        ClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        ClassificationTree.THREAD_WORK_LOAD = Integer.MAX_VALUE;
        BaggingClassification.MAX_THREADS = 5;
        BaggingClassification.SAMPLE_SIZE_COEF = 0.02;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 2);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            BaggingClassification baggingClassification = new BaggingClassification();
            baggingClassification.initialize(trainSet);
            String className = "model.supervised.cart.ClassificationTree";
            Evaluator evaluator = new ClassificationEvaluator();
            baggingClassification.baggingConfig(50, className, evaluator, trainSet);
            baggingClassification.train();

            evaluator.initialize(trainSet, baggingClassification);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, baggingClassification);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

            break;
        }

        // best: depth = 7, 42 round => 0.3357 on train 0.2967 on test

    }

    public static void eoec(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 44;
        int n = 11314;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
        EOECAdaBoost.ADABOOST_CLASSIFIER_CLASS_NAME = className;
        EOECAdaBoost.MAX_THREADS = 4;
        EOECAdaBoost.MAX_ITERATION = 10;
        EOECAdaBoost.DEFAULT_CODE_WORD_LENGTH = 25;
        DecisionStump.MAX_THREADS = 1;
        DecisionStump.THREAD_WORK_LOAD = Integer.MAX_VALUE;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 100);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            EOECAdaBoost eoecAdaBoost = new EOECAdaBoost();
            eoecAdaBoost.initialize(trainSet);
            eoecAdaBoost.train();

            Evaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(trainSet, eoecAdaBoost);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, eoecAdaBoost);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

            break;
        }

    }

    public static void mixtureGDATest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 44;
        int n = 11314;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        MixtureGaussianEM.MAX_ROUND = 50;
        MixtureGaussianEM.PRINT_GAP = 50;
        MixtureGaussianEM.THRESHOLD = 0.0001;
        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 3;
        MixtureGaussianDiscriminantAnalysis.MAX_THREADS = 4;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 2);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !trainIndexes.contains(x);
            int[] testIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());

            MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();
            mixGDA.initialize(trainSet);
            mixGDA.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(trainSet, mixGDA);
            evaluator.getPredictLabel();
            log.info("train accu: {}", evaluator.evaluate());

            DataSet testSet = dataset.subDataSetByRow(testIndexes);
            evaluator.initialize(testSet, mixGDA);
            evaluator.getPredictLabel();
            log.info("test accu: {}", evaluator.evaluate());

            break;
        }
    }

    public static void main(String[] args) throws Exception{
        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.expand.txt";
//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.full.no.x.y.txt";
//        String path = args[0];
//        int boost = Integer.parseInt(args[1]);
//
//       classificationTreeTest(path);
//
//        adaBoostTest(path);

//        adaBoostTest2(path);

//        eoec(path);

//        gradientBoostTest(path, 5);

        bagging(path);

//        neuralNetworkTest(path);

//        mixtureGDATest(path);
    }
}
