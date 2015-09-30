package model.supervised.linearregression;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.io.IOException;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class Main {

    public static void normEquaSpamTest () throws IOException {

        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        Evaluator eva = new Evaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, null);
        NormalEquation normalEquation= new NormalEquation();
        crossEvaluator.crossValidateEvaluate(normalEquation);

    }

    public static void normEquaHouseTest () throws IOException {

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/house.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
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
        NormalEquation normalEquation = new NormalEquation();
        crossEvaluator.crossValidateEvaluate(normalEquation);
    }


    public static void main(String[] args) throws IOException {

        NormalEquation.LAMBDA = 0.3;

        normEquaSpamTest ();
        System.out.println("\n\n\n");
        normEquaHouseTest();
    }
}
