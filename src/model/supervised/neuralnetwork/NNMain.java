package model.supervised.neuralnetwork;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import model.supervised.linearmodel.LogisticGradientDecent;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

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


//        System.out.println(dataset.getFeatureLength());
//        System.out.println(dataset.getInstanceLength());
//        System.out.println(dataset.getLabel(0));
//
//        Perceptron perceptron = new Perceptron();
//        perceptron.initialize(dataset);
//        perceptron.train();

        Evaluator eva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        Perceptron perceptron = new Perceptron();
        crossEvaluator.crossValidateEvaluate(perceptron);



    }

    public static void main(String[] args) throws Exception{

        Perceptron.ALPHA = 0.0025;
        perceptronTest();
    }
}
