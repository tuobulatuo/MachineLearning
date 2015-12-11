package model.supervised.knn;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.Norm;
import model.supervised.kernels.GaussianK;
import model.supervised.kernels.PolynomialK;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

import java.io.BufferedWriter;
import java.io.FileWriter;


/**
 * Created by hanxuan on 12/10/15 for machine_learning.
 */
public class KNNMain {

    private static Logger log = LogManager.getLogger(KNNMain.class);

    public static void spamTest (double k, String kernelClassName, EstimateBy estimateMethod, SelectNeighborBy selectNeighborMethod) throws Exception {

        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        ClassificationEvaluator eva = new ClassificationEvaluator();
        ClassificationEvaluator.ROC = false;
        ClassificationEvaluator.CONFUSION_MATRIX = true;
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);


        KNN.R = k;
        KNN.ESTIMATE_BY = estimateMethod;
        KNN.SELECT_NEIGHBOR_BY = selectNeighborMethod;
        KNN knn = new KNN(kernelClassName);
        crossEvaluator.crossValidateEvaluate(knn);
    }

    public static void digitTest (double k, String kernelClassName, EstimateBy estimateMethod, SelectNeighborBy selectNeighborMethod) throws Exception {

        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/digits.mnist/text/train.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 200;
        int n = 1200;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet train = builder.getDataSet();
        train.meanVarianceNorm();
        int[][] kFoldIndex = CrossValidationEvaluator.partition(train, 10);
        DataSet miniTrain = train.subDataSetByRow(kFoldIndex[0]);

        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/digits.mnist/text/test.txt";
        builder =
                new FullMatrixDataSetBuilder(path2, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);
        builder.build();
        DataSet test = builder.getDataSet();
        test.meanVarianceNorm();

        KNN.R = k;
        KNN.ESTIMATE_BY = estimateMethod;
        KNN.SELECT_NEIGHBOR_BY = selectNeighborMethod;
        KNN knn = new KNN(kernelClassName);
        knn.initialize(miniTrain);
        knn.train();

        ClassificationEvaluator eva = new ClassificationEvaluator();

        eva.initialize(miniTrain, knn);
        eva.getPredictLabelByProbs();
        log.info("KNN accu on train {}", eva.evaluate());
        log.info("zero neighbor count {}", knn.getZeroNeighborCounter());

        eva.initialize(test, knn);
        eva.getPredictLabelByProbs();
        log.info("KNN accu on test {}", eva.evaluate());
        log.info("zero neighbor count {}", knn.getZeroNeighborCounter());
    }

    public static void topFeatureTest(double k, String kernelClassName, EstimateBy estimateMethod, SelectNeighborBy selectNeighborMethod)
    throws Exception{

        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        KNN.R = k;
        KNN.ESTIMATE_BY = estimateMethod;
        KNN.SELECT_NEIGHBOR_BY = selectNeighborMethod;
        KNN knn = new KNN(kernelClassName);
        knn.initialize(dataset);

        int topK = 5;

        int[] topFeatureIndex = knn.TopFeatureIndex(topK);

        log.info("{}", topFeatureIndex);

        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.top5feature.data";
        BufferedWriter writer = new BufferedWriter(new FileWriter(path2), 1024 * 1024 * 32);
        for (int i = 0; i < dataset.getInstanceLength(); i++) {

            double[] X = dataset.getInstance(i);
            double y = dataset.getLabel(i);

            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < topFeatureIndex.length; j++) sb.append((float) X[topFeatureIndex[j]] + sep);
            sb.append(y);
            writer.write(sb.toString() + "\n");
        }
        writer.close();


        builder =
                new FullMatrixDataSetBuilder(path2, sep, hasHeader, needBias, topK, n, featureCategoryIndex, classification);
        builder.build();
        DataSet compactSet = builder.getDataSet();

        ClassificationEvaluator eva = new ClassificationEvaluator();
        ClassificationEvaluator.ROC = false;
        ClassificationEvaluator.CONFUSION_MATRIX = true;
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, compactSet, 10, Norm.MEANSD);

        KNN.R = 7;
        KNN knn2 = new KNN(kernelClassName);
        crossEvaluator.crossValidateEvaluate(knn2);
    }


    public static void main(String[] args) throws Exception{

        String kn1 = "model.supervised.kernels.EuclideanK";
        String kn2 = "model.supervised.kernels.GaussianK";
        String kn3 = "model.supervised.kernels.CosineK";
        String kn4 = "model.supervised.kernels.PolynomialK";

        spamTest(1, kn1, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK); //AVG_TEST (0.9152173913043479), AVG_TRAIN (0.9994204298478628)
        spamTest(3, kn1, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK); //AVG_TEST (0.912608695652174), AVG_TRAIN (0.9516058922965467)
        spamTest(7, kn1, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK); //AVG_TEST (0.9069565217391304), AVG_TRAIN (0.9278918135716012)

        spamTest(2.5, kn1, EstimateBy.NEIGHBOR, SelectNeighborBy.RADIUS);   //AVG_TEST (0.7452173913043479), AVG_TRAIN (0.9534894952909923)
        spamTest(3.5, kn1, EstimateBy.NEIGHBOR, SelectNeighborBy.RADIUS);   //AVG_TEST (0.788695652173913), AVG_TRAIN (0.9034532721564841)
        spamTest(4.5, kn1, EstimateBy.NEIGHBOR, SelectNeighborBy.RADIUS);   //AVG_TEST (0.7913043478260868), AVG_TRAIN (0.8561458584882878)

        GaussianK.GAMMA = 2;    //AVG_TEST (0.9065217391304348), AVG_TRAIN (0.9773243177976335)
        GaussianK.GAMMA = 2.5;  //AVG_TEST (0.9128260869565217), AVG_TRAIN (0.9899541173629558)
        GaussianK.GAMMA = 3;    //AVG_TEST (0.9163043478260869), AVG_TRAIN (0.9933832407631005)
        GaussianK.GAMMA = 10;   //AVG_TEST (0.9204347826086957), AVG_TRAIN (0.9974643805843998)
        spamTest(0, kn2, EstimateBy.DENSITY, SelectNeighborBy.RADIUS);

        GaussianK.GAMMA = 3;
        digitTest(1, kn2, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.9337 ~ 10% data
        digitTest(3, kn2, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.893
        digitTest(7, kn2, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.892

        digitTest(1, kn3, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.888
        digitTest(3, kn3, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.904
        digitTest(7, kn3, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.898

        digitTest(1, kn4, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.619
        digitTest(3, kn4, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.683
        digitTest(7, kn4, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK);  // 0.712

        digitTest(-0.66, kn3, EstimateBy.NEIGHBOR, SelectNeighborBy.RADIUS);  // 0.87

        GaussianK.GAMMA = 2.5;    //0.89
        digitTest(-1, kn2, EstimateBy.DENSITY, SelectNeighborBy.RANK);

        PolynomialK.GAMMA = 0.05;   // 0.74
        digitTest(-1, kn4, EstimateBy.DENSITY, SelectNeighborBy.RANK);

        topFeatureTest(3, kn2, EstimateBy.NEIGHBOR, SelectNeighborBy.RANK); // AVG_TEST (0.8319565217391306), AVG_TRAIN (0.8727119053368751)
    }
}
