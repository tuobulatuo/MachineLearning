package model.supervised.perceptron;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;

/**
 * Created by hanxuan on 12/11/15 for machine_learning.
 */
public class PerceptronMain {

    public static void perceptronTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/perceptron.data.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 4;
        int n = 1000;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.shiftCompressNorm();
        dataset.meanVarianceNorm();
        Perceptron.ALPHA = 1;
        Perceptron.PRINT_GAP = 1;
        Perceptron.BUCKET_COUNT = 1;
        Perceptron.COST_DECENT_THRESHOLD = 0.000000001;
        Perceptron.MAX_ROUND = 100;
        Perceptron perceptron = new Perceptron();
        perceptron.initialize(dataset);
        perceptron.train();

    }


    public static void dualPerceptronTest(String kn) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/perceptron.data.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 4;
        int n = 1000;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.shiftCompressNorm();
        dataset.meanVarianceNorm();

        DualPerceptron.PRINT_GAP = 1;
        DualPerceptron.BUCKET_COUNT = 1;
        DualPerceptron.COST_DECENT_THRESHOLD = 0.000000001;
        DualPerceptron.MAX_ROUND = 100;

        DualPerceptron dualPerceptron = new DualPerceptron(kn);
        dualPerceptron.initialize(dataset);
        dualPerceptron.train();

    }

    public static void dualPerceptronTest2(String kn) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/twoSpirals.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 2;
        int n = 1000;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.shiftCompressNorm();
        dataset.meanVarianceNorm();

        DualPerceptron.PRINT_GAP = 1;
        DualPerceptron.BUCKET_COUNT = 1;
        DualPerceptron.COST_DECENT_THRESHOLD = 0.000000001;
        DualPerceptron.MAX_ROUND = 100;

        DualPerceptron dualPerceptron = new DualPerceptron(kn);
        dualPerceptron.initialize(dataset);
        dualPerceptron.train();

    }

    public static void main(String[] args) throws Exception{

        perceptronTest();

        String kn1 = "model.supervised.kernels.EuclideanK";
        String kn2 = "model.supervised.kernels.GaussianK";
        String kn3 = "model.supervised.kernels.CosineK";
        String kn4 = "model.supervised.kernels.PolynomialK";
        String kn5 = "model.supervised.kernels.LinearK";


//        dualPerceptronTest2(kn5);
//        dualPerceptronTest2(kn1);
        dualPerceptronTest2(kn2);
//        dualPerceptronTest2(kn3);
//        dualPerceptronTest2(kn4);

    }
}
