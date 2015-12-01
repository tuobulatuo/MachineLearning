package model.supervised.ecoc;

import data.DataSet;
import data.builder.Builder;
import data.builder.SparseMatrixDataSetBuilder;
import model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump;
import model.supervised.svm.SVMsSMO;
import model.supervised.svm.kernels.GaussianK;
import performance.ClassificationEvaluator;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class ECOCMain {

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
        ECOCAdaBoost.ADABOOST_CLASSIFIER_CLASS_NAME = className;
        ECOCAdaBoost.MAX_THREADS = 1;
        ECOCAdaBoost.MAX_ITERATION = 200;
        DecisionStump.MAX_THREADS = 1;

        ECOCAdaBoost eoecAdaBoost = new ECOCAdaBoost();
        eoecAdaBoost.initialize(trainSet);
        eoecAdaBoost.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, eoecAdaBoost);
        evaluator.getPredictLabelByProbs();
        System.out.print(evaluator.evaluate());
    }

    public static void newsgroupSVMTest() throws Exception{

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

        SVMsSMO.C = 0.05;
//        SVMsSMO.MAX_CHANGE = 100000;
        SVMsSMO.MAX_CHANGE = Integer.MAX_VALUE;
        SVMsSMO.LRU_MAX_ENTRY = 200;
        SVMsSMO.DEBUG = false;
        SVMsSMO.PRINT_GAP = 20000;
        SVMsSMO.TOL = 0.1;
        SVMsSMO.EPS1 = 0.001;
        SVMsSMO.EPS2 = 1E-8;

//        String kernelClassName = "model.supervised.svm.kernels.GaussianK";
//        GaussianK.GAMMA = 1 / (double) 200;

        String kernelClassName = "model.supervised.svm.kernels.LinearK";
        ECOCSVMs.KERNEL_CLASS_NAME = kernelClassName;
        ECOCSVMs.MAX_THREADS = 4;
        ECOCSVMs.DEFAULT_CODE_WORD_LENGTH = 20;

        ClassificationEvaluator.THREAD_WORK_LOAD = 125;

        ECOCSVMs ecocsvMs = new ECOCSVMs();
        ecocsvMs.initialize(trainSet);
        ecocsvMs.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, ecocsvMs);
        evaluator.getPredictLabelByProbs();
        System.out.print(evaluator.evaluate());
    }

    public static void main(String[] args) throws Exception{
//        newsgroupTest(); // accu 0.73
        newsgroupSVMTest(); // accu 0.86
    }

}
