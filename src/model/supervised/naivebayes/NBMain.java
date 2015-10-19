package model.supervised.naivebayes;

import algorithms.parameterestimate.MixtureGaussianEM;
import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;


/**
 * Created by hanxuan on 10/15/15 for machine_learning.
 */
public class NBMain {

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

        MixtureGaussianEM.THRESHOLD = 1E-300;
        MixtureGaussianEM.MAX_ROUND = 1000;
        MixtureGaussianEM.PRINT_GAP = 200;

        MixtureGaussian.COMPONENTS = 3;
        MixtureGaussian.MAX_THREADS = 2;
        MixtureGaussian mixtureGaussianNB = new MixtureGaussian();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.POS = 1;
        ClassificationEvaluator.CONFUSION_MATRIX = true;
        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.NULL);
        crossEvaluator.crossValidateEvaluate(mixtureGaussianNB);

    }

    public static void main(String[] args) throws Exception{

//        multinoulliNBTest(2, new double[]{80});
//        System.out.println("\n\n\n");
//        multinoulliNBTest(4, new double[]{25, 50, 75});
//        System.out.println("\n\n\n");
//        multinoulliNBTest(9, new double[]{11, 22, 33, 44, 55, 66, 77, 88});
//
//        gaussianNBTest();


        mixGaussianNBTest();
    }
}
