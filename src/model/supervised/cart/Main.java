package model.supervised.cart;

import data.DataSet;
import data.builder.FullMatrixDataSetBuilder;
import data.builder.Builder;
import data.core.Norm;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

import java.io.IOException;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class Main {

    public static void classificationTest() throws IOException {

        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/homework/hw1/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path2, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        ClassificationEvaluator classifyEva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(classifyEva, dataset, 10, Norm.MINMAX);

        Tree classificationTree = new ClassificationTree();
        classificationTree.MAX_DEPTH = 10;
        crossEvaluator.crossValidateEvaluate(classificationTree);

    }

    public static void main(String[] args) throws IOException {
        classificationTest();
    }
}
