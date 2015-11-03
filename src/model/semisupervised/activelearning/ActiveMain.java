package model.semisupervised.activelearning;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.adaboot.SAMME;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class ActiveMain {

    public static void activeAdaBoostTest() throws Exception{

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

        SAMME.NEED_ROUND_REPORT = false;
        ActiveAdaBoost.PERCENT_START = 0.002;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 3);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            ActiveAdaBoost activeAdaBoost = new ActiveAdaBoost(10, className, new ClassificationEvaluator(), testSet);
            activeAdaBoost.initialize(trainSet);
            activeAdaBoost.train();
            break;
        }
    }

    public static void main(String[] args) throws Exception{
        activeAdaBoostTest();
    }
}
