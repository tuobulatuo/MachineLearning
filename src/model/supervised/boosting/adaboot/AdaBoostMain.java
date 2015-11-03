package model.supervised.boosting.adaboot;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.builder.SparseMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree;
import model.supervised.boosting.adaboot.adaboostclassifier.WeightedClassificationTree;
import org.neu.util.rand.RandomUtils;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

import java.util.Arrays;
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
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            samme.boostConfig(50, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            evaluator.getArea();
            evaluator.printROC();

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
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.RandomDecisionStump";
            samme.boostConfig(200, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            evaluator.getArea();
            evaluator.printROC();

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
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.WeightedClassificationTree";
            samme.boostConfig(20, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            System.out.print(evaluator.evaluate());

            break;
        }
    }

    public static void SAMMESampleSimulationTest() throws Exception{

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
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree";
            sammeSampleSimulation.boostConfig(50, className, new ClassificationEvaluator(), testSet);
            sammeSampleSimulation.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, sammeSampleSimulation);
            evaluator.getPredictLabel();
            System.out.print(evaluator.evaluate());

            break;
        }
    }

    public static void newsgroupTest() throws Exception{
        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/train.trec/feature_matrix.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 1754;
        int n = 11314;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new SparseMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet trainSet = builder.getDataSet();

        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/test.trec/feature_matrix.txt";
        builder =
                new SparseMatrixDataSetBuilder(path2, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet testSet = builder.getDataSet();

        SAMMESampleSimulation.NEED_ROUND_REPORT = true;
        SAMMESampleSimulation.SAMPLE_SIZE_COEF = 1;
        AdaBoostClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        AdaBoostClassificationTree.MAX_DEPTH = 100;
        AdaBoostClassificationTree.MAX_THREADS = 4;

        SAMMESampleSimulation sammeSampleSimulation = new SAMMESampleSimulation();
        sammeSampleSimulation.initialize(trainSet);
        String className = "model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree";
        sammeSampleSimulation.boostConfig(100, className, new ClassificationEvaluator(), testSet);
        sammeSampleSimulation.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, sammeSampleSimulation);
        evaluator.getPredictLabel();
        System.out.print(evaluator.evaluate());
    }

    public static void voteTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/vote/vote.data";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 434;
        int[] featureCategoryIndex = RandomUtils.getIndexes(m);
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
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            samme.boostConfig(50, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            evaluator.getArea();
            evaluator.printROC();

            break;
        }
    }

    public static void crxTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/crx/crx1.data.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 100;

//                A1:	b, a.
//                A2:	continuous.
//                A3:	continuous.
//                A4:	u, y, l, t.
//                A5:	g, p, gg.
//                A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
//                A7:	v, h, bb, j, n, z, dd, ff, o.
//                A8:	continuous.
//                A9:	t, f.
//                A10:	t, f.
//                A11:	continuous.
//                A12:	t, f.
//                A13:	g, p, s.
//                A14:	continuous.
//                A15:	continuous.
//                A16: +,-         (class attribute)

        int[] featureCategoryIndex = {0, 3, 4, 5, 6, 8, 9, 11, 12};
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
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            samme.boostConfig(100, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabel();
            evaluator.getArea();
            evaluator.printROC();

            break;
        }
    }

    public static void main(String[] args) throws Exception{

//        DecisionStumpTest();

//        RandomDecisionStumpTest();

//        SAMMETest();

        SAMMESampleSimulationTest();

//        newsgroupTest();

//        voteTest();

//        crxTest();
    }
}
