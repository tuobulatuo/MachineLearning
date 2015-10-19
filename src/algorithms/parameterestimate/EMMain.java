package algorithms.parameterestimate;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import model.supervised.generative.MixtureGaussianDiscriminantAnalysis;

/**
 * Created by hanxuan on 10/18/15 for machine_learning.
 */
public class EMMain {

    public static void twoMixtureGaussianTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/2gaussian.em.txt";;
        String sep = " ";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 2;
        int n = 4700;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        MixtureGaussianEM.MAX_ROUND = 500;
        MixtureGaussianEM.PRINT_GAP = 10;
        MixtureGaussianEM.THRESHOLD = 1E-10;

        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 2;
        MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();
        mixGDA.initialize(dataset);
        mixGDA.train();

    }

    public static void threeMixtureGaussianTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/3gaussian.em.txt";;
        String sep = " ";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 2;
        int n = 4700;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        MixtureGaussianEM.MAX_ROUND = 500;
        MixtureGaussianEM.PRINT_GAP = 10;
        MixtureGaussianEM.THRESHOLD = 1E-10;

        MixtureGaussianDiscriminantAnalysis.COMPONENTS = 3;
        MixtureGaussianDiscriminantAnalysis mixGDA = new MixtureGaussianDiscriminantAnalysis();
        mixGDA.initialize(dataset);
        mixGDA.train();

    }

    public static void main(String[] args) throws Exception{
        twoMixtureGaussianTest();
        threeMixtureGaussianTest();
    }
}
