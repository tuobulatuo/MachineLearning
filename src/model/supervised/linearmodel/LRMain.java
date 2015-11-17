package model.supervised.linearmodel;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;
import utils.random.RandomUtils;

import java.io.IOException;
import java.util.stream.IntStream;

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
        LogisticGradientDecent.LAMBDA = 0.0;
        LogisticGradientDecent.ALPHA = 0.00005;
        LogisticGradientDecent.MAX_ROUND = 8000;
        ClassificationEvaluator.ROC = true;

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        LogisticGradientDecent lg = new LogisticGradientDecent();
        crossEvaluator.crossValidateEvaluate(lg);

    }

    public static void lgPollutedSetTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spam_polluted/allSet";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 1057;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int trainSize = 4140;
        int allSize = 4601;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, allSize).toArray());

        LogisticGradientDecent.BUCKET_COUNT = 10;
        LogisticGradientDecent.LAMBDA = 0.001;
        LogisticGradientDecent.ALPHA = 0.01;
        LogisticGradientDecent.MAX_ROUND = 2000;

        LogisticGradientDecent logisticGradientDecent = new LogisticGradientDecent();
        logisticGradientDecent.initialize(trainSet);
        logisticGradientDecent.train();

        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator.ROC = false;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainSet, logisticGradientDecent);
        evaluator.getPredictLabel();
        evaluator.evaluate();

        evaluator.initialize(testSet, logisticGradientDecent);
        evaluator.getPredictLabel();
        evaluator.evaluate();
    }


    public static void main(String[] args) throws Exception {

//        normEquaHouseTest();
//        lmHouseTest();

//        normEquaSpamTest();
//        lmSpamTest();
//        lgSpamTest();

        lgPollutedSetTest(); // 0.92 ~ 0.93
    }
}
