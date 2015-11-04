package model.supervised.bagging;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.Evaluator;
import utils.random.RandomUtils;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public abstract class BootstrapAveraging implements Trainable, Predictable, Bagging {

    public static int SAMPLE_SIZE_COEF = 1;

    public static int MAX_THREADS = 4;

    private static Logger log = LogManager.getLogger(BootstrapAveraging.class);

    protected DataSet trainData = null;

    protected DataSet testData = null;

    protected Evaluator roundEvaluator = null;

    protected Trainable[] trainables = null;

    protected Predictable[] predictables = null;

    protected int[] indexes = null;

    private double[] probs = null;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    @Override
    public void train() {

        service = Executors.newFixedThreadPool(MAX_THREADS);
        countDownLatch = new CountDownLatch(trainables.length);

        IntStream.range(0, trainables.length).forEach(i -> service.submit(() -> {
            try {
                doTask(i);
            }catch (Throwable t) {
                log.error(t.getMessage(), t);
            }
            countDownLatch.countDown();
        }));
        try {
            TimeUnit.SECONDS.sleep(10);
            countDownLatch.await();
        }catch (Throwable t) {
            log.error(t.getMessage(), t);
        }
        service.shutdown();

        log.info("BootstrapAveraging training finished ...");
    }

    @Override
    public void initialize(DataSet d) {
        this.trainData = d;
        indexes = RandomUtils.getIndexes(d.getInstanceLength());
        probs = new double[indexes.length];
        Arrays.fill(probs, 1 / (double) indexes.length);
    }

    @Override
    public void baggingConfig(int iteration, String trainableClassName, Evaluator evaluator, DataSet testData) {

        trainables = new Trainable[iteration];
        try {
            for (int i = 0; i < iteration; i++) {
                trainables[i] = (Trainable) Class.forName(trainableClassName).getConstructor().newInstance();
            }
        }catch (Exception e) {
            log.error(e.getMessage(), e);
        }
        predictables = new Predictable[iteration];
        roundEvaluator = evaluator;
        this.testData = testData;
    }

    private void doTask(int taskId) {

        EnumeratedIntegerDistribution integerDistribution = new EnumeratedIntegerDistribution(indexes, probs);
        int[] trainingIndexes = integerDistribution.sample(indexes.length * SAMPLE_SIZE_COEF);
        DataSet taskData = trainData.subDataSetByRow(trainingIndexes);
        trainables[taskId].initialize(taskData);
        trainables[taskId].train();
        predictables[taskId] = trainables[taskId].offer();

        log.info("BootstrapAveraging task {}/{} finished...", taskId, trainables.length);

    }
}
