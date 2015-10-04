package model.supervised.neuralnetwork;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import model.supervised.linearmodel.LogisticGradientDecent;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

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

        MultilayerPerceptron.PRINT_HIDDEN = false;
        for (int i = 0; i < dataset.getInstanceLength(); i++) {
            double[] X = dataset.getInstance(i);
            System.out.println(Arrays.toString(X) + " => " + Arrays.toString(nn.yVector((int)nn.predict(X))));
        }
    }

    public static void neuralNetworkDigitsTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/digits.Xy.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 400;
        int n = 5000;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        int[] structure = {401, 51, 26, 10};
        boolean biased = true;
        MultilayerPerceptron.BUCKET_COUNT = 10;
        MultilayerPerceptron.ALPHA = 0.00125;
        MultilayerPerceptron.LAMBDA = 0.00000;
        MultilayerPerceptron.COST_DECENT_THRESHOLD = 0;
        MultilayerPerceptron.MAX_ROUND = 3000;
        MultilayerPerceptron.PRINT_HIDDEN = false;
        MultilayerPerceptron.EPSILON = 0.0001;
        MultilayerPerceptron nn = new MultilayerPerceptron(structure, biased);

        // 0.91 accu

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, null);
        crossEvaluator.crossValidateEvaluate(nn);
    }

    public static void neuralNetworkSpam() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";;
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 57;
        int n = 4700;
        int[] featureCategoryIndex = {};
        boolean classification = false;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        int[] structure = {58, 15, 2};
        boolean biased = true;
        MultilayerPerceptron.BUCKET_COUNT = 1;
        MultilayerPerceptron.ALPHA = 0.00001;
        MultilayerPerceptron.LAMBDA = 0.00000;
        MultilayerPerceptron.COST_DECENT_THRESHOLD = 0;
        MultilayerPerceptron.MAX_ROUND = 8000;
        MultilayerPerceptron.PRINT_GAP = 50;
        MultilayerPerceptron.PRINT_HIDDEN = false;
        MultilayerPerceptron.EPSILON = 0.002;
        MultilayerPerceptron nn = new MultilayerPerceptron(structure, biased);

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        crossEvaluator.crossValidateEvaluate(nn);
    }

    public static void main(String[] args) throws Exception{

//        perceptronTest();

//        neuralNetworkTest();

//        neuralNetworkDigitsTest();

        neuralNetworkSpam();
    }
}
