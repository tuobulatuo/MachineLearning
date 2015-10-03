package model.supervised.neuralnetwork;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;

import java.util.Arrays;


/**
 * Created by hanxuan on 10/1/15 for machine_learning.
 */
public class NNMain {


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
        dataset.meanVarianceNorm();
        Perceptron.ALPHA = 0.0025;
        Perceptron perceptron = new Perceptron();
        perceptron.initialize(dataset);
        perceptron.train();

    }


    public static void neuralNetworkTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/nn.data";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 8;
        int n = 8;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        int[] structure = {8, 3, 8};
        boolean biased = false;
        MultilayerPerceptron.BUCKET_COUNT = 1;
        MultilayerPerceptron nn = new MultilayerPerceptron(structure, biased);
        nn.initialize(dataset);
        nn.train();

        for (int i = 0; i < dataset.getInstanceLength(); i++) {
            double[] X = dataset.getInstance(i);
            System.out.println(Arrays.toString(X) + " => " + Arrays.toString(nn.yVector((int)nn.predict(X))));
        }
    }

    public static void main(String[] args) throws Exception{

//        perceptronTest();

        neuralNetworkTest();
    }
}
