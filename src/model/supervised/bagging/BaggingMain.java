package model.supervised.bagging;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.cart.ClassificationTree;
import model.supervised.cart.RegressionTree;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class BaggingMain {

    public static void regressionTest() throws Exception {

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/house.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 13;
        int n = 507;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        RegressionTree.MAX_DEPTH = 5;
        RegressionTree.MAX_THREADS = 1;
        RegressionTree.COST_DROP_THRESHOLD = 0;

        double avgMse = 0;
        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            BaggingRegression baggingRegression = new BaggingRegression();
            baggingRegression.initialize(trainSet);
            String className = "model.supervised.cart.RegressionTree";
            Evaluator evaluator = new Evaluator();
            baggingRegression.baggingConfig(50, className, evaluator, testSet);
            baggingRegression.train();

            evaluator.initialize(testSet, baggingRegression);
            evaluator.getPredictLabel();
            avgMse += evaluator.evaluate();
            System.out.println(avgMse);
            break;
        }
        avgMse /= (double) 10;
        System.out.println(avgMse);

    }

    public static void classificationTest() throws Exception{
        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();
        DataSet dataset = builder.getDataSet();

        ClassificationTree.MAX_DEPTH = 5;
        ClassificationTree.MAX_THREADS = 1;
        ClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;

        double avgError = 0;
        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            BaggingClassification baggingClassification = new BaggingClassification();
            baggingClassification.initialize(trainSet);
            String className = "model.supervised.cart.ClassificationTree";
            Evaluator evaluator = new ClassificationEvaluator();
            baggingClassification.baggingConfig(50, className, evaluator, testSet);
            baggingClassification.train();

            evaluator.initialize(testSet, baggingClassification);
            evaluator.getPredictLabel();
            avgError += 1 - evaluator.evaluate();
            System.out.println(avgError);
            break;
        }
        avgError /= (double) 10;
        System.out.println(avgError);

    }

    public static void main(String[] args) throws Exception{
        regressionTest();
        classificationTest();
    }
}
