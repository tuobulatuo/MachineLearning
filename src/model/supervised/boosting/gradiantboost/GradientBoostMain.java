package model.supervised.boosting.gradiantboost;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.builder.SparseMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.adaboot.SAMMESampleSimulation;
import model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class GradientBoostMain {

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

        GradientRegressionTree.MAX_DEPTH = 5;
        GradientBoostRegression.NEED_REPORT = true;
        GradientBoostRegression.OUTLIER_QUANTILE = 90;

        double avgMse = 0;
        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            GradientBoostRegression boostRegression = new GradientBoostRegression();
            boostRegression.initialize(trainSet);
            String className = "model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree";
            Evaluator evaluator = new Evaluator();
            boostRegression.boostConfig(100, className, evaluator, testSet);
            boostRegression.train();

            evaluator.initialize(testSet, boostRegression);
            evaluator.getPredictLabel();
            avgMse += evaluator.evaluate();
            System.out.println(avgMse);
            break;
        }
        avgMse /= (double) 10;
        System.out.println(avgMse);

    }

    public static void classificationTest() throws Exception{
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

        GradientBoostClassification.NEED_REPORT = true;
        GradientBoostClassification.MAX_THREADS = 4;
        GradientRegressionTree.MAX_THREADS = 1;
        GradientRegressionTree.MAX_DEPTH = 5;
        GradientBoostRegression.OUTLIER_QUANTILE = 100;

        GradientBoostClassification boostClassification = new GradientBoostClassification();
        boostClassification.initialize(trainSet);
        String className = "model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree";
        boostClassification.boostConfig(4, className, new Evaluator(), testSet);
        boostClassification.train();
    }

    public static void main(String[] args) throws Exception{
        regressionTest();
        classificationTest(); // accu 0.79
    }
}
