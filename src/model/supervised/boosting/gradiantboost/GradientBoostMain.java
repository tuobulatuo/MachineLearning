package model.supervised.boosting.gradiantboost;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
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

        GradientRegressionTree.MAX_DEPTH = 3;
        GradientBoostRegression.NEED_REPORT = true;

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
            boostRegression.boostConfig(20, className, evaluator, testSet);
            boostRegression.train();

            evaluator.initialize(testSet, boostRegression);
            evaluator.getPredictLabel();
            System.out.print(evaluator.evaluate());
            break;
        }
    }

    public static void main(String[] args) throws Exception{
        regressionTest();
    }
}
