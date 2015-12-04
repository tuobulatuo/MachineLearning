package model.supervised.boosting.gradiantboost;

import data.DataSet;
import data.core.AMatrix;
import data.core.Label;
import gnu.trove.list.array.TIntArrayList;
import model.Predictable;
import model.Trainable;
import model.supervised.boosting.Boost;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientBooster;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.Evaluator;
import utils.array.ArraySumUtil;
import utils.array.ArrayUtil;
import utils.random.RandomUtils;
import utils.sort.SortIntDoubleUtils;

import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/30/15 for machine_learning.
 */
public class GradientBoostClassificationV2 implements Predictable, Trainable, Boost {

    public static boolean NEED_REPORT = false;

    public static double LEARNING_RATE = 0.1;

    public static double SAMPLE_RATE = 0.8;

    public static int MAX_THREADS = 4;

    private static Logger log = LogManager.getLogger(GradientBoostClassification.class);

    private GradientBooster[][] boosters = null;

    private float[][] scoreCache = null;

    private DataSet trainData = null;

    private TIntArrayList indices = null;

    private DataSet testData = null;

    private DataSet[] roundData = null;

    private int classCount = -1;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    private ClassificationEvaluator roundEvaluator = null;

    private float[] roundTestAccu = null;

    private float[] roundTrainAccu = null;

    private float[] roundKL = null;

    private boolean[] roundIndicator = null;

    private float[][] tempLabels = null;


    @Override
    public double predict(double[] feature) {
        double[] probs = probs(feature);
        int[] index = RandomUtils.getIndexes(classCount);
        SortIntDoubleUtils.sort(index, probs);
        return index[index.length - 1];
    }

    @Override
    public double[] probs(double[] feature) {

        double[] probs = new double[classCount];
        for (int i = 0; i < boosters.length; i++){
            if (roundIndicator[i]){
                for (int j = 0; j < boosters[i].length; j++)
                    probs[j] += LEARNING_RATE * boosters[i][j].boostPredict(feature);
            }
        }

        IntStream.range(0, classCount).forEach(i -> probs[i] = Math.exp(probs[i]));

        ArraySumUtil.normalize(probs);

        return probs;
    }

    @Override
    public void train() {

        for (int i = 0; i < boosters.length; i++) {

            long t1 = System.currentTimeMillis();

            final int ROUND = i;

            service = Executors.newFixedThreadPool(MAX_THREADS);
            countDownLatch = new CountDownLatch(classCount);
            IntStream.range(0, classCount).forEach(j -> service.submit(() -> {
                long tic = System.currentTimeMillis();

                try {
                    boosters[ROUND][j].boostInitialize(roundData[j], "");
                    boosters[ROUND][j].boost();
                }catch (Throwable t) {
                    log.error(t.getMessage(), t);
                }

                long toc = System.currentTimeMillis();
                log.debug("round {}, task: {}/{} finished, elapsed {} ms", ROUND, j, classCount, toc - tic);
                countDownLatch.countDown();
            }));
            try {
                TimeUnit.SECONDS.sleep(10);
                countDownLatch.await();
            }catch (Throwable t) {
                log.error(t.getMessage(), t);
            }
            service.shutdown();

            roundIndicator[ROUND] = true;

            long t2 = System.currentTimeMillis();

            float[] probs = new float[classCount];
            for (int j = 0; j < trainData.getInstanceLength(); j++) {

                double[] x = trainData.getInstance(j);
                double y = trainData.getLabel(j);
                float[] ys = new float[classCount];
                ys[(int)y] = 1;

                for (int k = 0; k < classCount; k++) {
                    scoreCache[j][k] += LEARNING_RATE * boosters[ROUND][k].boostPredict(x);
                    probs[k] = (float) Math.exp(scoreCache[j][k]);
                }

                ArraySumUtil.normalize(probs);

                roundKL[ROUND] += (float) ArrayUtil.KLDivergence(ys, probs);

                for (int k = 0; k < classCount; k++) tempLabels[k][j] = ys[k] - probs[k];
            }

            roundKL[ROUND] /= trainData.getInstanceLength();

            indices.shuffle(new Random());
            int[] indexArray = indices.toArray();
            TIntArrayList sample = new TIntArrayList((int) (indexArray.length * SAMPLE_RATE));
            for (int j = 0; j < indexArray.length * SAMPLE_RATE; j++) sample.add(indexArray[j]);
            int[] sampleArray = sample.toArray();
            AMatrix m = trainData.getFeatureMatrix().subMatrixByRow(sampleArray);
            for (int j = 0; j < classCount; j++) {
                roundData[j] = new DataSet(m, new Label(tempLabels[j], null).subLabelByRow(sampleArray));
            }

            long t3 = System.currentTimeMillis();

            if (NEED_REPORT) {
                statisticReport(ROUND);
            }

            long t4 = System.currentTimeMillis();

            log.info("round {}, KL {}", ROUND, roundKL[ROUND]);
            log.info("boost {} |update gradient {} |report {}| total {}", t2-t1, t3-t2, t4-t3, t4-t1);
        }

        log.info("GradientBoostClassification training finished ...");

        if (NEED_REPORT) {
            printRoundReport();
        }
    }

    @Override
    public void printRoundReport() {
        log.info("================ round report ================");
        log.info("roundTestAccu {}", roundTestAccu);
        log.info("roundTrainAccu {}", roundTrainAccu);
        log.info("roundKL {}", roundKL);
        log.info("================ ============ ================");
    }

    @Override
    public void initialize(DataSet d) {

        trainData = d;
        indices = new TIntArrayList(RandomUtils.getIndexes(d.getInstanceLength()));
        classCount = trainData.getLabels().getClassIndexMap().size();
        scoreCache = new float[trainData.getInstanceLength()][classCount];
        roundData = new DataSet[classCount];
        tempLabels = new float[classCount][trainData.getInstanceLength()];

        for (int i = 0; i < classCount; i++) {
            roundData[i] = new DataSet(trainData.getFeatureMatrix(), makeClassLabel(i));
        }

        log.info("initialize finished, tempLabels + ScoreCache MEM use ~ {} GB",
                2.0 * scoreCache.length * scoreCache[0].length * 4 / 1024 /1024 / 1024);
    }

    @Override
    public void boostConfig(int iteration, String boosterClassName, Evaluator evaluator, DataSet testData) throws
            Exception {

        GradientBooster[][] boosters = new GradientBooster[iteration][classCount];
        for (int i = 0; i < iteration; i++)
            for (int j = 0; j < classCount; j++)
                boosters[i][j] = (GradientBooster) Class.forName(boosterClassName).getConstructor().newInstance();

        this.boosters = boosters;

        roundTestAccu = new float[iteration];
        roundTrainAccu = new float[iteration];
        roundKL = new float[iteration];
        roundIndicator = new boolean[iteration];

        roundEvaluator = (ClassificationEvaluator) evaluator;
        this.testData = testData;

        log.info("GradientBoostRegression configTrainable: ");
        log.info("boosters count: {}", boosters.length);
        log.info("boosters CLASS: {}", boosterClassName);
    }

    private Label makeClassLabel(int classId) {

        float[] labels = tempLabels[classId];
        for (int i = 0; i < labels.length; i++) {
            if (trainData.getLabel(i) == classId) labels[i] = 1.0F;
            labels[i] -= 1 / classCount;
        }
        return new Label(labels, trainData.getLabels().getClassIndexMap());
    }

    private void statisticReport(int round){

        roundEvaluator.initialize(trainData, this);
        roundEvaluator.getPredictLabelByProbs();
        roundTrainAccu[round] = (float) roundEvaluator.evaluate();

        roundEvaluator.initialize(testData, this);
        roundEvaluator.getPredictLabelByProbs();
        roundTestAccu[round] = (float) roundEvaluator.evaluate();
    }
}
