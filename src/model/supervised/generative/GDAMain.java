package model.supervised.generative;

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
public class GDAMain {

    public static void gdaTest() throws Exception{

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

        GaussianDiscriminantAnalysis.COV_DISTINCT = false;
        GaussianDiscriminantAnalysis gda = new GaussianDiscriminantAnalysis();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.NULL);
        crossEvaluator.crossValidateEvaluate(gda);

    }

    public static void mixtureGDATest() throws Exception{

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

        MixtureGaussianEM.MAX_ROUND = 500;
        MixtureGaussianEM.PRINT_GAP = 1;
        MixtureGaussianEM.THRESHOLD = 0.5;

        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 25;
        MixtureGaussianDiscriminantAnalysis.MAX_THREADS = 2;
        MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.CONFUSION_MATRIX = false;
        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.NULL);
        crossEvaluator.crossValidateEvaluate(mixGDA);

    }

    public static void main(String[] args) throws Exception{

        gdaTest();

        mixtureGDATest();
    }
}
