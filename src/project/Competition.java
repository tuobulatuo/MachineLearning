package project;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import model.supervised.neuralnetwork.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import utils.random.RandomUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/8/15 for machine_learning.
 */
public class Competition {

    private static Logger log = LogManager.getLogger(Competition.class);

    public static void neuralNetworkTest(String path) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 51;
        int n = 877982;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4, 5, 6, 7};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int[] structure = {137, 22, 39};
        boolean biased = true;
        NeuralNetwork.MAX_THREADS = 7;
        NeuralNetwork.THREAD_WORK_LOAD = 500;
        NeuralNetwork.BUCKET_COUNT = 439;
        NeuralNetwork.ALPHA = 0.06 / (double) NeuralNetwork.BUCKET_COUNT;

        NeuralNetwork.COST_DECENT_THRESHOLD = 0;
        NeuralNetwork.MAX_ROUND = 4000;
        NeuralNetwork.PRINT_GAP = 4000;
        NeuralNetwork.PRINT_HIDDEN = false;
        NeuralNetwork.EPSILON = 0.0001;

        int trainSize = 877982;
        int testSize = 884262;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        NeuralNetwork nn = new NeuralNetwork(structure, biased);
        nn.initialize(trainSet);
        nn.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainSet, nn);
        evaluator.getPredictLabel();
        log.info("train accu: {}", evaluator.evaluate());

        double accu = 0;
        for (int j = 0; j < trainSet.getInstanceLength(); j++) {
            double y = trainSet.getLabel(j);
            double[] yVector = new double[structure[structure.length - 1]];
            yVector[(int) y] = 1;

            double[] feature = trainSet.getInstance(j);
            double[] probs = nn.probs(feature);

            for (int k = 0; k < yVector.length; k++) {
                accu += - yVector[k] * Math.log(probs[k]);
            }
        }

        log.info("trainSet avg loss {}", accu / (double) trainSet.getInstanceLength());


        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());
        String probsPredictsPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.probs.predicts.txt";
        BufferedWriter writer = new BufferedWriter(new FileWriter(probsPredictsPath), 1024 * 1024 * 32);

        String head = "Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS";
        writer.write(head+"\n");

        Map<String, Integer> idxMap = new HashMap<>();
        String[] es = head.split(",");
        for (int i = 1; i < es.length; i++) {
            idxMap.put(es[i], i - 1);
        }

        Map<Integer, Object> indexClassMap = dataset.getLabels().getIndexClassMap();

        int id = 0;
        for (int i = 0; i < testSet.getInstanceLength(); i++) {

            StringBuilder sb = new StringBuilder(id + ",");

            double[] probs = nn.probs(testSet.getInstance(i));
            double[] arrangedProbs = new double[probs.length];
            for (int j = 0; j < probs.length; j++) {
                String className = (String) indexClassMap.get(j);
                int arrangeIndex = idxMap.get(className);
                arrangedProbs[arrangeIndex] = probs[j];
            }

            Arrays.stream(arrangedProbs).forEach(x -> sb.append(x + ","));
            sb.deleteCharAt(sb.length() - 1);
            writer.write(sb.toString() + "\n");
            id ++;
        }
        writer.close();
    }

    public static void main(String[] args) throws Exception{
        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.all.txt";
        neuralNetworkTest(path);
    }
}
