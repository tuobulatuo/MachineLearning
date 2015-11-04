package project;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.adaboot.SAMMESampleSimulation;
import model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree;
import model.supervised.boosting.gradiantboost.GradientBoostClassification;
import model.supervised.boosting.gradiantboost.GradientBoostRegression;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree;
import model.supervised.cart.ClassificationTree;
import model.supervised.cart.Tree;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import performance.Evaluator;

import java.io.IOException;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class Analysis {

    public static void classificationTreeTest(String path) throws IOException {

//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.hour.txt";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 5;
        int n = 87000;
        int[] featureCategoryIndex = {0, 1, 2};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        ClassificationEvaluator.ROC = true;
        ClassificationEvaluator.CONFUSION_MATRIX = false;
        ClassificationEvaluator classifyEva = new ClassificationEvaluator();
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(classifyEva, dataset, 10, Norm.NULL);

        Tree classificationTree = new ClassificationTree();
        ClassificationTree.MAX_THREADS = 2;
        ClassificationTree.THREAD_WORK_LOAD = 1;
        ClassificationTree.MAX_DEPTH = 25;
        ClassificationTree.INFORMATION_GAIN_THRESHOLD = 0.01;
        ClassificationTree.MIN_INSTANCE_COUNT = 20;
        crossEvaluator.crossValidateEvaluate(classificationTree);

    }

    public static void gradientBoostTest(String path, int boost) throws Exception{
//        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.hour.txt";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 5;
        int n = 87000;
        int[] featureCategoryIndex = {0, 1, 2};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet trainSet = builder.getDataSet();

        GradientBoostClassification.NEED_REPORT = true;
        GradientBoostClassification.MAX_THREADS = 2;
        GradientRegressionTree.MAX_THREADS = 1;
        GradientRegressionTree.MAX_DEPTH = 10;
        GradientBoostRegression.OUTLIER_QUANTILE = 100;

        GradientBoostClassification boostClassification = new GradientBoostClassification();
        boostClassification.initialize(trainSet);
        String className = "model.supervised.boosting.gradiantboost.gradientboostor.GradientRegressionTree";
        boostClassification.boostConfig(boost, className, new Evaluator(), trainSet);
        boostClassification.train();
    }

    public static void adaBoostTest(String path) throws Exception{
//        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/train.trec/feature_matrix.txt";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 5;
        int n = 11314;
        int[] featureCategoryIndex = {0,1,2};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMMESampleSimulation.NEED_ROUND_REPORT = true;
        SAMMESampleSimulation.SAMPLE_SIZE_COEF = 1;
        AdaBoostClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        AdaBoostClassificationTree.MAX_DEPTH = 25;
        AdaBoostClassificationTree.MAX_THREADS = 2;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMMESampleSimulation sammeSampleSimulation = new SAMMESampleSimulation();
            sammeSampleSimulation.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree";
            sammeSampleSimulation.boostConfig(3, className, new ClassificationEvaluator(), testSet);
            sammeSampleSimulation.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, sammeSampleSimulation);
            evaluator.getPredictLabel();
            System.out.print(evaluator.evaluate());

            break;
        }

    }

    public static void main(String[] args) throws Exception{
//        String path = args[0];
//        int boost = Integer.parseInt(args[1]);
//        classificationTreeTest(path);
//        gradientBoostTest(path, boost);
        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.hour.txt";
        adaBoostTest(path);
    }
}
