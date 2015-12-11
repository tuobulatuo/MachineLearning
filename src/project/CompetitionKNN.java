package project;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.kernels.GaussianK;
import model.supervised.knn.EstimateBy;
import model.supervised.knn.KNN;
import model.supervised.knn.SelectNeighborBy;
import model.supervised.neuralnetwork.NeuralNetwork;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
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
public class CompetitionKNN {

    private static Logger log = LogManager.getLogger(CompetitionKNN.class);

    public static void knnTest(String inPath, String OutPath, int partition, String kn, String selectNeighbor,
    String estimate, double r, int threads, int threadsWorkLoad, int validateSize, boolean test) throws Exception{

        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 52;
        int n = 1762311;
        int[] featureCategoryIndex = {0,1,2,3,4,5,6,7,8};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(inPath, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();
        System.gc();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int trainSize = 878049;
        int testSize = 884262;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));

        int[][] kFoldIndex = CrossValidationEvaluator.partition(trainSet, partition);
        DataSet miniTrainSet = dataset.subDataSetByRow(kFoldIndex[0]);

        KNN.ESTIMATE_BY = EstimateBy.valueOf(estimate);
        KNN.SELECT_NEIGHBOR_BY = SelectNeighborBy.valueOf(selectNeighbor);
        KNN.R = r;

        KNN knn = new KNN(kn);
        knn.initialize(miniTrainSet);
        knn.train();

        ClassificationEvaluator.THREAD_WORK_LOAD = threadsWorkLoad;
        ClassificationEvaluator.MAX_THREADS = threads;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(miniTrainSet, knn);
        evaluator.getPredictLabelByProbs();
        log.info("miniTrainSet accu: {}", evaluator.evaluate());
        log.info("miniTrainSet log loss {}", evaluator.logLoss());


        TIntHashSet validateIndex = new TIntHashSet(trainSize);
        validateSize = Math.min(validateSize, kFoldIndex.length);
        IntStream.range(1, validateSize).forEach(i -> validateIndex.addAll(kFoldIndex[i]));
        DataSet validateSet = trainSet.subDataSetByRow(validateIndex.toArray());
        evaluator.initialize(validateSet, knn);
        evaluator.getPredictLabelByProbs();
        log.info("validate accu: {}", evaluator.evaluate());
        log.info("validate log loss: {}", evaluator.logLoss());

        if (test) {

            DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, trainSize + testSize).toArray());

            evaluator.initialize(testSet, knn);
            evaluator.getPredictLabelByProbs();
            double[][] probs = evaluator.getProbs();

            BufferedWriter writer = new BufferedWriter(new FileWriter(OutPath), 1024 * 1024 * 64);

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

                double[] probsI = probs[i];
                double[] arrangedProbs = new double[probsI.length];
                for (int j = 0; j < probsI.length; j++) {
                    String className = (String) indexClassMap.get(j);
                    int arrangeIndex = idxMap.get(className);
                    arrangedProbs[arrangeIndex] = probsI[j];
                }

                Arrays.stream(arrangedProbs).forEach(x -> sb.append((float) x + ","));
                sb.deleteCharAt(sb.length() - 1);
                writer.write(sb.toString() + "\n");

                if (id ++ % 100000 == 0) log.info("write {} lines ..", id);
            }
            writer.close();
        }


        log.info("task done, exit ..");
    }


    public static void main(String[] args) throws Exception{


//        String inPath = args[0];
//        String outPath = args[1];
//        int partition = Integer.parseInt(args[2]);
//        String kn = args[3];
//        String selectNeighbor = args[4];
//        String estimate = args[5];
//        double r = Double.parseDouble(args[6]);
//        int threads = Integer.parseInt(args[7]);
//        double kernelParam = Double.parseDouble(args[8]);
//        int threadWorkLoad = Integer.parseInt(args[9]);
//        int validateSize = Integer.parseInt(args[10]);
//        boolean test = Boolean.parseBoolean(args[11]);


        String kn1 = "model.supervised.kernels.EuclideanK";
        String kn2 = "model.supervised.kernels.GaussianK";
        String kn3 = "model.supervised.kernels.CosineK";
        String kn4 = "model.supervised.kernels.PolynomialK";

        String inPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/nobuzz/data.all.expand.txt";
        String outPath = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.probs.predicts.txt";
        int partition = 50;
        String kn = kn2;
        String selectNeighbor = "RANK";
        String estimate = "DENSITY";
        double r = 1;
        int threads = 4;
        double kernelParam = 2.5;
        int threadWorkLoad = 100;
        int validateSize = 2;
        boolean test = false;

        GaussianK.GAMMA = kernelParam;

//        String kn1 = "model.supervised.kernels.EuclideanK";
//        String kn2 = "model.supervised.kernels.GaussianK";
//        String kn3 = "model.supervised.kernels.CosineK";
//        String kn4 = "model.supervised.kernels.PolynomialK";

        knnTest(inPath, outPath, partition, kn, selectNeighbor, estimate, r, threads, threadWorkLoad, validateSize, test);
    }
}
