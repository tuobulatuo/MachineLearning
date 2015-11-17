package feature.haar;

import data.DataSet;
import data.core.FullMatrix;
import data.core.Label;
import model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump;
import model.supervised.ecoc.ECOCAdaBoost;
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

        ecoc(train, test);
    }
}
