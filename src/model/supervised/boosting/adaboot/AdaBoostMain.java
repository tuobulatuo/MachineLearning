package model.supervised.boosting.adaboot;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/31/15 for machine_learning.
 */
public class AdaBoostMain {

    public static void DecisionStumpTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = true;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.DecisionStump";
            samme.boostConfig(20, className, new ClassificationEvaluator(), testSet);
            samme.train();

            break;
        }
    }


    public static void RandomDecisionStumpTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = true;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.RandomDecisionStump";
            samme.boostConfig(20, className, new ClassificationEvaluator(), testSet);
            samme.train();

            break;
        }
    }

    public static void SAMMETest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/letter-recognition.reformat3.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 20000;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = false;
        WeightedClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        WeightedClassificationTree.MAX_DEPTH = 100;
        WeightedClassificationTree.MAX_THREADS = 1;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.WeightedClassificationTree";
            samme.boostConfig(20, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            System.out.print(evaluator.evaluate());

            break;
        }
    }

    public static void SAMMESampleSimulatinTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/letter-recognition.reformat3.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 20000;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMMESampleSimulation.NEED_ROUND_REPORT = true;
        SAMMESampleSimulation.SAMPLE_SIZE_COEF = 1;
        AdaBoostClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        AdaBoostClassificationTree.MAX_DEPTH = 100;
        AdaBoostClassificationTree.MAX_THREADS = 1;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMMESampleSimulation sammeSampleSimulation = new SAMMESampleSimulation();
            sammeSampleSimulation.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.AdaBoostClassificationTree";
            sammeSampleSimulation.boostConfig(20, className, new ClassificationEvaluator(), testSet);
            sammeSampleSimulation.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, sammeSampleSimulation);
            evaluator.getPredictLabel();
            System.out.print(evaluator.evaluate());

            break;
        }
    }

    public static void main(String[] args) throws Exception{

//        DecisionStumpTest();
//        RandomDecisionStumpTest();

//        SAMMETest();

        SAMMESampleSimulatinTest();
    }
}
