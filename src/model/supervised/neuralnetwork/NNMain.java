package model.supervised.neuralnetwork;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
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
        NeuralNetwork.BUCKET_COUNT = 1;
        NeuralNetwork.MAX_ROUND = 40000;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.PRINT_GAP = 10000;
        NeuralNetwork.ALPHA = 0.01;
        NeuralNetwork.LAMBDA = 0.001;
        NeuralNetwork.EPSILON = 0.1;
        NeuralNetwork nn = new NeuralNetwork(structure, biased);
        nn.initialize(dataset);
        nn.train();

        NeuralNetwork.PRINT_HIDDEN = true;
        for (int i = 0; i < dataset.getInstanceLength(); i++) {
            double[] X = dataset.getInstance(i);
            System.out.println(Arrays.toString(X) + " | " + Arrays.toString(nn.yVector((int)nn.predict(X))));
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

        int[] structure = {401, 26, 10};
        boolean biased = true;
        NeuralNetwork.BUCKET_COUNT = 1;
        NeuralNetwork.ALPHA = 0.001 / NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.LAMBDA = 0.001 / NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 5000;
        NeuralNetwork.PRINT_GAP = 10;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.0001;
        NeuralNetwork nn = new NeuralNetwork(structure, biased);

        // 0.91 accu

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
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
        NeuralNetwork.BUCKET_COUNT = 5;
        NeuralNetwork.THREAD_WORK_LOAD = 200;
        NeuralNetwork.ALPHA = 0.0001 / (double) NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.LAMBDA = 0.00001 / (double) NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 8000;
        NeuralNetwork.PRINT_GAP = 50;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.002;
        NeuralNetwork nn = new NeuralNetwork(structure, biased);

        ClassificationEvaluator.ROC = true;

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        crossEvaluator.crossValidateEvaluate(nn);
    }

    public static void letterTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/letter-recognition.reformat3.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 16;
        int n = 20000;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        int[] structure = {17, 52, 26};
        boolean biased = true;
        NeuralNetwork.BUCKET_COUNT = 10;
        // best alpha = 0.02 + {17, 50, 26} => 0.912 test 0.945 train
        // best alpha = 0.01 + {17, 52, 26} => 0.922 test 0.939 train
        NeuralNetwork.ALPHA = 0.015 / (double) NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.THREAD_WORK_LOAD = 200;
        NeuralNetwork.LAMBDA = 0.00001 / (double) NeuralNetwork.BUCKET_COUNT;
        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 8000;
        NeuralNetwork.MAX_THREADS = 4;
        NeuralNetwork.PRINT_GAP = 100;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.002;
        NeuralNetwork nn = new NeuralNetwork(structure, biased);

        ClassificationEvaluator.ROC = false;

        ClassificationEvaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 20, Norm.MEANSD);
        crossEvaluator.crossValidateEvaluate(nn);


    }

    public static void main(String[] args) throws Exception{

        perceptronTest();

//        neuralNetworkTest();

//        neuralNetworkDigitsTest();
//
//        neuralNetworkSpam();

//        letterTest();
    }
}
