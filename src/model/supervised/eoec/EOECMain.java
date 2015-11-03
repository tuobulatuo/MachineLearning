package model.supervised.eoec;

import data.DataSet;
import data.builder.Builder;
import data.builder.SparseMatrixDataSetBuilder;
import model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump;
import performance.ClassificationEvaluator;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class EOECMain {

    public static void newsgroupTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/train.trec/feature_matrix.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 1754;
        int n = 11314;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new SparseMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet trainSet = builder.getDataSet();

        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/test.trec/feature_matrix.txt";
        builder =
                new SparseMatrixDataSetBuilder(path2, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet testSet = builder.getDataSet();

        String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
        EOECAdaBoost.ADABOOST_CLASSIFIER_CLASS_NAME = className;
        EOECAdaBoost.MAX_THREADS = 1;
        EOECAdaBoost.MAX_ITERATION = 200;
        DecisionStump.MAX_THREADS = 1;

        EOECAdaBoost eoecAdaBoost = new EOECAdaBoost();
        eoecAdaBoost.initialize(trainSet);
        eoecAdaBoost.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, eoecAdaBoost);
        evaluator.getPredictLabel();
        System.out.print(evaluator.evaluate());
    }

    public static void main(String[] args) throws Exception{
        newsgroupTest();
    }

}
