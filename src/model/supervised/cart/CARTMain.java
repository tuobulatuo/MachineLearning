package model.supervised.cart;

import data.DataSet;
import data.builder.FullMatrixDataSetBuilder;
import data.builder.Builder;
import data.core.Norm;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.io.IOException;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class CARTMain {

    public static void classificationTest() throws IOException {

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

        ClassificationEvaluator classifyEva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(classifyEva, dataset, 10, null);

        Tree classificationTree = new ClassificationTree();
        ClassificationTree.MAX_DEPTH = 9;
        ClassificationTree.INFORMATION_GAIN_THRESHOLD = 0.001;
        ClassificationTree.MIN_INSTANCE_COUNT = 5;
        crossEvaluator.crossValidateEvaluate(classificationTree);

    }

    public static void regressionTest() throws IOException{
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

        Evaluator eva = new Evaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, null);

        RegressionTree regressionTree = new RegressionTree();
        RegressionTree.MAX_DEPTH = Integer.MAX_VALUE;
        RegressionTree.COST_DROP_THRESHOLD = 0;
        RegressionTree.MIN_INSTANCE_COUNT = 1;
        crossEvaluator.crossValidateEvaluate(regressionTree);

    }

    public static void main(String[] args) throws IOException {
        classificationTest();
//        regressionTest();
    }
}
