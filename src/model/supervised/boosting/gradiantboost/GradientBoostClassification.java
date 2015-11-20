package model.supervised.boosting.gradiantboost;

import data.DataSet;
import data.core.Label;
import model.Predictable;
import model.Trainable;
import model.supervised.boosting.Boost;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.Evaluator;
import utils.array.ArraySumUtil;
import utils.random.RandomUtils;
import utils.sort.SortIntDoubleUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class GradientBoostClassification implements Predictable, Trainable, Boost {

    public static boolean NEED_REPORT = false;

    public static int MAX_THREADS = 4;

    private static Logger log = LogManager.getLogger(GradientBoostClassification.class);

    private DataSet trainData = null;

    private DataSet testData = null;

    private GradientBoostRegression[] regressions = null;

    private int classCount = -1;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    private double kl = -1;

    private double testError = -1;

    private double trainError = -1;

    @Override
    public double predict(double[] feature) {
        double[] probs = probs(feature);
        int[] indexes = RandomUtils.getIndexes(classCount);
        SortIntDoubleUtils.sort(indexes, probs);
        return indexes[classCount - 1];
    }

    public double[] probs (double[] feature) {
        double[] probs = new double[classCount];
        for (int i = 0; i < classCount; i++) {
            probs[i] = regressions[i].predict(feature);
        }
        return ArraySumUtil.normalize(probs);
    }

    @Override
    public void train() {

        service = Executors.newFixedThreadPool(MAX_THREADS);
        countDownLatch = new CountDownLatch(classCount);
        IntStream.range(0, classCount).forEach(i -> service.submit(() -> {
            try {
                doTask(i);
            }catch (Throwable t) {
                log.error(t.getMessage(), t);
            }
            log.info("task: {}/{} finished...", i, classCount);
            countDownLatch.countDown();
        }));
        try {
            TimeUnit.SECONDS.sleep(10);
            countDownLatch.await();
        }catch (Throwable t) {
            log.error(t.getMessage(), t);
        }
        service.shutdown();

        log.info("GradientBoostClassification training finished ...");

        if(NEED_REPORT) {
            statisticReport();
            printRoundReport();
        }
    }

    @Override
    public void initialize(DataSet d) {
        this.trainData = d;
        classCount = d.getLabels().getIndexClassMap().size();
    }


    @Override
    public void boostConfig(int iteration, String boosterClassName, Evaluator evaluator, DataSet testData) throws Exception {
        regressions = new GradientBoostRegression[classCount];
        for (int i = 0; i < classCount; i++) {
            regressions[i] = new GradientBoostRegression();
            regressions[i].boostConfig(iteration, boosterClassName, evaluator, testData);
        }
        this.testData = testData;
        log.info("GradientBoostClassification config finished, classCount {}, regressor iteration {}", classCount, iteration);
    }

    @Override
    public void printRoundReport() {
        log.info("============ report ============");
        log.info("KL-divergence {}", kl);
        log.info("test error {}", testError);
        log.info("train error {}", trainError);
        log.info("============ ====== ============");
    }

    private void doTask(int classId) {

        Label classLabel = makeClassLabel(classId);
        DataSet classData = new DataSet(trainData.getFeatureMatrix(), classLabel);
        regressions[classId].initialize(classData);
        regressions[classId].train();
    }

    private Label makeClassLabel(int classId) {

        float[] labels = new float[trainData.getInstanceLength()];
        for (int i = 0; i < labels.length; i++) {
            if (trainData.getLabel(i) == classId) {
                labels[i] = 1.0F;
            }
        }
        return new Label(labels, trainData.getLabels().getClassIndexMap());
    }

    private void statisticReport() {

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(trainData, this);
        evaluator.getPredictLabelByProbs();
        trainError = 1 - evaluator.evaluate();

        evaluator.initialize(testData, this);
        evaluator.getPredictLabelByProbs();
        testError = 1 - evaluator.evaluate();
    }
}
