package model.supervised.naivebayes;

import algorithms.parameterestimate.MixtureGaussianEM;
import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import utils.random.RandomUtils;

import java.util.stream.IntStream;


/**
 * Created by hanxuan on 10/15/15 for machine_learning.
 */
public class NBMain {

    private static Logger log = LogManager.getLogger(NBMain.class);

    public static void multinoulliNBTest(int bins, double[] quotients) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";;
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4700;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        Multinoulli.BINS = bins;
        Multinoulli.QUOTIENTS = quotients;
        Multinoulli multinoulli = new Multinoulli();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.POS = 1;
        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.NULL);
        crossEvaluator.crossValidateEvaluate(multinoulli);

    }

    public static void gaussianNBTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";;
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4700;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        Gaussian g = new Gaussian();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.POS = 1;
        ClassificationEvaluator.CONFUSION_MATRIX = false;
        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.NULL);
        crossEvaluator.crossValidateEvaluate(g);

    }

    public static void mixGaussianNBTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";;
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4700;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        MixtureGaussianEM.THRESHOLD = 1E-5;
        MixtureGaussianEM.MAX_ROUND = 1000;
        MixtureGaussianEM.PRINT_GAP = 200;

        MixtureGaussian.COMPONENTS = 9;
        MixtureGaussian.MAX_THREADS = 2;
        MixtureGaussian mixtureGaussianNB = new MixtureGaussian();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.POS = 1;
        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.NULL);
        crossEvaluator.crossValidateEvaluate(mixtureGaussianNB);

    }

    public static void gaussianNBPollutedSetTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spam_polluted/allSet";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 1057;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        int trainSize = 4140;
        int allSize = 4601;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, allSize).toArray());
        Gaussian g = new Gaussian();
        g.initialize(trainSet);
        g.train();


        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainSet, g);
        evaluator.getPredictLabel();
        evaluator.evaluate();

        evaluator.initialize(testSet, g);
        evaluator.getPredictLabel();
        evaluator.evaluate();
    }

    public static void gaussianNBMissingSetTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spam_missing/missing.all.txt";
        String sep = "\t";
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

        int trainSize = 3681;
        int allSize = 4601;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, allSize).toArray());
        Multinoulli multinoulli = new Multinoulli();
        multinoulli.initialize(trainSet);
        multinoulli.train();


        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainSet, multinoulli);
        evaluator.getPredictLabel();
        evaluator.evaluate();

        evaluator.initialize(testSet, multinoulli);
        evaluator.getPredictLabel();
        evaluator.evaluate();
    }

    public static void gaussianNBPCAPollutedSetTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spam_polluted/allSet.pca.100";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 100;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        int trainSize = 4140;
        int allSize = 4601;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, allSize).toArray());
        Gaussian g = new Gaussian();
        g.initialize(trainSet);
        g.train();


        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainSet, g);
        evaluator.getPredictLabel();
        evaluator.evaluate();

        evaluator.initialize(testSet, g);
        evaluator.getPredictLabel();
        evaluator.evaluate();
    }

    public static void main(String[] args) throws Exception{

//        multinoulliNBTest(2, new double[]{50});
//        System.out.println("\n\n\n");
//        multinoulliNBTest(4, new double[]{25, 50, 75});
//        System.out.println("\n\n\n");
//        multinoulliNBTest(9, new double[]{11, 22, 33, 44, 55, 66, 77, 88});

//        gaussianNBTest();

//        mixGaussianNBTest();

//        gaussianNBPollutedSetTest();

//        gaussianNBPCAPollutedSetTest();

        gaussianNBMissingSetTest();
    }
}
