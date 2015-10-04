package model.supervised.linearmodel;

import algorithms.gradient.Decent;
import com.sun.prism.paint.Gradient;
import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.io.IOException;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class LRMain {

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

        NormalEquation.LAMBDA = 0.1;

        ClassificationEvaluator eva = new ClassificationEvaluator();
        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.EPSILON = 0.45;
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

    public static void lmHouseTest () throws IOException {

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

        LinearGradientDecent.BUCKET_COUNT = 5;
        LinearGradientDecent.LAMBDA = 0.0;

        Evaluator eva = new Evaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        LinearGradientDecent lm = new LinearGradientDecent();
        crossEvaluator.crossValidateEvaluate(lm);
    }

    public static void lmSpamTest () throws IOException {

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

        LinearGradientDecent.BUCKET_COUNT = 5;
        LinearGradientDecent.LAMBDA = 0.0;

        Evaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        LinearGradientDecent lm = new LinearGradientDecent();
        crossEvaluator.crossValidateEvaluate(lm);

    }

    public static void lgSpamTest () throws IOException {

        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        LogisticGradientDecent.BUCKET_COUNT = 100;
        LogisticGradientDecent.LAMBDA = 0.1;
        LogisticGradientDecent.ALPHA = 0.00005;
        ClassificationEvaluator.ROC = true;

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        LogisticGradientDecent lg = new LogisticGradientDecent();
        crossEvaluator.crossValidateEvaluate(lg);

    }


    public static void main(String[] args) throws IOException {

        normEquaHouseTest();
        lmHouseTest();

        normEquaSpamTest();
        lmSpamTest();
        lgSpamTest();
    }
}
