package feature.haar;

import data.DataSet;
import data.core.FullMatrix;
import data.core.Label;
import model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump;
import model.supervised.ecoc.ECOCAdaBoost;
import model.supervised.ecoc.ECOCSVMs;
import model.supervised.svm.SVMsSMO;
import model.supervised.kernels.GaussianK;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;

import java.util.HashMap;

/**
 * Created by hanxuan on 11/15/15 for machine_learning.
 */
public class HAARMain {

    private static Logger log = LogManager.getLogger(HAARMain.class);

    public static void ecoc(DataSet trainSet, DataSet testSet) {

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, 20);
        trainSet = trainSet.subDataSetByRow(kFoldIndex[0]);

        String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
        ECOCAdaBoost.ADABOOST_CLASSIFIER_CLASS_NAME = className;
        ECOCAdaBoost.MAX_THREADS = 2;
        ECOCAdaBoost.MAX_ITERATION = 100;
        ECOCAdaBoost.DEFAULT_CODE_WORD_LENGTH = 20;

        DecisionStump.MAX_THREADS = 1;
        DecisionStump.THREAD_WORK_LOAD = Integer.MAX_VALUE;

        ECOCAdaBoost eoecAdaBoost = new ECOCAdaBoost();
        eoecAdaBoost.initialize(trainSet);
        eoecAdaBoost.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, eoecAdaBoost);
        evaluator.getPredictLabel();
        log.info("test accu {}", evaluator.evaluate());

        /*
        * 5% data + 100 round + 20 length => 0.8866 | 0.8948
        * */

    }

    public static void ecocSVM(DataSet trainSet, DataSet testSet) throws Exception{

        trainSet.meanVarianceNorm();
        testSet.meanVarianceNorm();

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, 50);
        trainSet = trainSet.subDataSetByRow(kFoldIndex[0]);

        SVMsSMO.C = 10;
        SVMsSMO.MAX_CHANGE = 100000;
        SVMsSMO.PRINT_GAP = 20000;
        SVMsSMO.LRU_MAX_ENTRY = 2000000;
        SVMsSMO.TOL = 0.01;
        SVMsSMO.EPS1 = 0.001;
        SVMsSMO.EPS2 = 1E-8;

        String kernelClassName = "model.supervised.kernels.GaussianK";
        GaussianK.GAMMA = 1 / (double) 200;

//        String kernelClassName = "model.supervised.kernels.LinearK";
        ECOCSVMs.KERNEL_CLASS_NAME = kernelClassName;
        ECOCSVMs.MAX_THREADS = 4;
        ECOCSVMs.DEFAULT_CODE_WORD_LENGTH = 20;

        ClassificationEvaluator.THREAD_WORK_LOAD = 125;

        ECOCSVMs ecocsvMs = new ECOCSVMs();
        ecocsvMs.initialize(trainSet);
        ecocsvMs.train();

        int[][] testIndexes = CrossValidationEvaluator.partition(testSet, 10);
        testSet = testSet.subDataSetByRow(testIndexes[0]);

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainSet, ecocsvMs);
        evaluator.getPredictLabelByProbs();
        log.info("train accu {}", evaluator.evaluate());

        evaluator.initialize(testSet, ecocsvMs);
        evaluator.getPredictLabelByProbs();
        log.info("test accu {}", evaluator.evaluate());

        /*
        * 5% data + 100000 max change + 30 length + C = 0.01, TOL = 0.01  => 0.82
        *
        *
        * 1% data + 10 length + C = 10, GaussianKernel(gamma = 1 / 200) =>
        * train accu 0.9966666666666667
        * test accu 0.935
        * */

    }

    public static DataSet buildSet(String path, String imageFile, String labelFile) throws Exception{

        MNISTReader reader = new MNISTReader(path, imageFile, labelFile);

        int total = reader.getTotal();
        int colNum = reader.getNumCols();
        int rowNum = reader.getNumRows();

        log.info("{} * {} * {} = {} Mbytes", total, colNum, rowNum, total * colNum * rowNum * 4 / 1024 / 1024);


        float[][] featureMatrix = new float[total][];
        float[] labelVec = new float[total];
        int counter = 0;
        while (reader.hasNext()) {
            int[][] image = new int[rowNum][colNum];
            int label = reader.readNext(image);

            labelVec[counter] = label;
            HAARExtractor extractor = new HAARExtractor(image);
            featureMatrix[counter] = extractor.extract(10, 10, 2, 2);

            if (counter ++ % 2000 == 0) log.info("{} images processed ..", counter);
        }

        FullMatrix matrix = new FullMatrix(featureMatrix, new boolean[featureMatrix.length]);
        HashMap<Object, Integer> classIndexMap = new HashMap<>(total);
        for (int i = 0; i < 10; i++) classIndexMap.put(i, i);
        Label labels = new Label(labelVec, classIndexMap);

        return new DataSet(matrix, labels);
    }

//    public static void writeTxt(DataSet data, String filePath) throws Exception{
//
//        BufferedWriter writer = new BufferedWriter(new FileWriter(filePath), 1024 * 1024 * 64);
//        for (int i = 0; i < data.getInstanceLength(); i++) {
//            double[] x = data.getInstance(i);
//            double y = data.getLabel(i);
//            StringBuilder sb = new StringBuilder();
//            for (double e: x) sb.append(e+"\t");
//            sb.append(y);
//            writer.write(sb.toString() + "\n");
//        }
//        writer.close();
//    }



    public static void main(String[] args) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/digits.mnist";

        String labelTestFile = "t10k-labels-idx1-ubyte";
        String imageTestFile = "t10k-images-idx3-ubyte";

        String labelTrainFile = "train-labels-idx1-ubyte";
        String imageTrainFile = "train-images-idx3-ubyte";

        DataSet train = buildSet(path, imageTrainFile, labelTrainFile);
        DataSet test = buildSet(path, imageTestFile, labelTestFile);

        log.info("train dim {} {}", train.getInstanceLength(), train.getFeatureLength());
        log.info("test dim {} {}", test.getInstanceLength(), test.getFeatureLength());

//        ecoc(train, test);

        ecocSVM(train, test);
    }
}
