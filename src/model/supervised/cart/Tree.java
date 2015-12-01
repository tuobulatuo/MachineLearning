package model.supervised.cart;

import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.random.RandomUtils;
import utils.sort.SortIntDoubleUtils;
//import org.neu.util.sort.SortIntDoubleUtils;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public abstract class Tree implements Trainable, Predictable{

    private static final Logger log = LogManager.getLogger(Tree.class);

    public static int TREE_ID = 0;

    public static int MAX_DEPTH = 10;

    public static int MIN_INSTANCE_COUNT = 5;

    public static int MAX_THREADS = 4;

    public static int THREAD_WORK_LOAD = 1;

    public static double SELECTED_FEATURE_LENGTH = Integer.MAX_VALUE;

    protected int td;

    protected Tree left = null;

    protected Tree right = null;

    protected double meanResponse = 0;

    protected int depth = 0;

    protected int[] existIds = null;

    protected DataSet dataSet = null;

    protected int featureId = Integer.MIN_VALUE;

    protected double featureThreshold = Integer.MAX_VALUE;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    public Tree() {}

    public Tree(int depth, DataSet dataSet, int[] existIds) {

        this.depth = depth;
        this.dataSet = dataSet;
        this.existIds = existIds;
        td = TREE_ID ++;
    }


    protected abstract double gainByCriteria(double[] labels, int position, int[] sortedIds);

    protected abstract boolean lessThanImpurityGainThreshold(double gain);

    protected abstract void setTreeLabel();

    protected abstract void split(int[] leftGroup, int[] rightGroup);

    @Override
    public double predict(double[] feature) {

        if (left == null) {
            return meanResponse;
        }else if(feature[featureId] < featureThreshold) {
            return this.left.predict(feature);
        }else {
            return this.right.predict(feature);
        }
    }

    @Override
    public void train() {

        if (pure()) {
            log.debug("[STOP GROW] pure node {} ...", td);
            setTreeLabel();
            return;
        }

        if (stopGrow()) {
            log.debug("[STOP GROW] depth({}) >= MAX_DEPTH({}) || existIds.length({}) <= MIN_INSTANCE_COUNT({})",
                    depth, MAX_DEPTH, existIds.length, MIN_INSTANCE_COUNT);
            setTreeLabel();
            return;
        }

        // random forest
        final int featureLength = (int) Math.min(dataSet.getFeatureLength(), SELECTED_FEATURE_LENGTH);
        TIntArrayList selectedFeatureIndices = new TIntArrayList(RandomUtils.getIndexes(dataSet.getFeatureLength()));
        selectedFeatureIndices.shuffle(new Random());
        int[] selectedFeature = new int[featureLength];
        for (int i = 0; i < featureLength; i++) selectedFeature[i] = selectedFeatureIndices.get(i);

        final AtomicInteger bestFeatureId = new AtomicInteger(Integer.MIN_VALUE);
        final AtomicDouble bestThreshold = new AtomicDouble(Integer.MIN_VALUE);
        final AtomicDouble bestGain = new AtomicDouble(Integer.MIN_VALUE);

        service = Executors.newFixedThreadPool(MAX_THREADS);

        log.debug("featureLength {}, existIds {}", featureLength, existIds.length);

        int realTaskCount = (int) Math.ceil(featureLength / (double) THREAD_WORK_LOAD);
        countDownLatch = new CountDownLatch(realTaskCount);
        log.debug("Task Count: {}", countDownLatch.getCount());
        TIntArrayList taskPackage = new TIntArrayList();
        Arrays.stream(selectedFeature).forEach(i -> {
            taskPackage.add(i);
            if (taskPackage.size() == THREAD_WORK_LOAD || i == featureLength - 1) {
                TIntArrayList taskPackage2 = new TIntArrayList(taskPackage);
                service.submit(()->
                {
                    try {

                        int currentBestFeatureId = -1;
                        double currentBestGain = Integer.MIN_VALUE;
                        double currentBestThreshold = Integer.MIN_VALUE;
                        for(int taskId : taskPackage2.toArray()) {
                            double[] report = doTask(taskId);
                            if (report[0] > currentBestGain) {
                                currentBestFeatureId = taskId;
                                currentBestGain = report[0];
                                currentBestThreshold = report[1];
                            }
                        }
                        if (currentBestGain > bestGain.get()) {
                            bestGain.getAndSet(currentBestGain);
                            bestFeatureId.getAndSet(currentBestFeatureId);
                            bestThreshold.getAndSet(currentBestThreshold);
                        }
                        log.debug("one package processed/{}", realTaskCount);
                    }catch (Throwable t) {
                        log.error(t.getMessage(), t);
                    }
                    countDownLatch.countDown();
                });
                taskPackage.clear();
            }
        });

        try {
            TimeUnit.MILLISECONDS.sleep(10);
            countDownLatch.await();
        } catch (InterruptedException e) {
            log.error(e.getMessage(), e);
        }
        service.shutdown();

        log.debug("All task finished, service shutdown ...");

        log.debug("Best FeatureId: {}, threshold: {}, gain: {}", bestFeatureId.get(), bestThreshold.get(), bestGain.get());

        if (lessThanImpurityGainThreshold(bestGain.get())) {

            log.debug("[STOP GROW] lessThanImpurityGainThreshold, best gain: {}", bestGain.get());
            setTreeLabel();
            return;
        }

        featureId = bestFeatureId.get();
        featureThreshold = bestThreshold.get();

        if(featureId == Integer.MIN_VALUE) {
            log.warn("featureId == Integer.MIN_VALUE ~ now stop growing ...");
            setTreeLabel();
            return;
        }

        grow();
    }

    @Override
    public Predictable offer() {
        return this;
    }

    private void grow() {

        TIntHashSet left = new TIntHashSet();
        TIntHashSet right = new TIntHashSet();
        for (int i : existIds) {
            if (dataSet.getEntry(i, featureId) < featureThreshold) {
                left.add(i);
            }else {
                right.add(i);
            }
        }

        if (left.size() == 0 || right.size() == 0) {
            setTreeLabel();
            return;
        }

        split(left.toArray(), right.toArray());

        this.left.train();
        this.right.train();

        log.debug("[TREE NODE] {}", td);

    }

    private void sortFeatureLabel(int[] ids, double[] labelArray, double[] featureArray, int featureId) {
        IntStream.range(0, ids.length).forEach(i -> featureArray[i] = dataSet.getEntry(ids[i], featureId));
        SortIntDoubleUtils.sort(ids, featureArray);
        IntStream.range(0, ids.length).forEach(i -> labelArray[i] = dataSet.getLabel(ids[i]));
    }


    private boolean stopGrow() {

        return depth >= MAX_DEPTH || existIds.length <= MIN_INSTANCE_COUNT;
    }

    private boolean pure() {

        if (existIds.length == 0) return true;

        double first = dataSet.getLabel(existIds[0]);
        IntPredicate pred = (i) -> dataSet.getLabel(i) == first;
        return Arrays.stream(existIds).allMatch(pred);
    }

    private double[] doTask(int featureId) {

        int[] ids = existIds.clone();
        double[] features = new double[ids.length];
        double[] labels = new double[ids.length];
        sortFeatureLabel(ids, labels, features, featureId);

        int pointer = 1;
        int loopCounter = 1;
        double bestGain = Integer.MIN_VALUE;
        double bestThreshold = Integer.MIN_VALUE;
        while (pointer < ids.length) {
            if (features[pointer] == features[pointer - 1]) {
                ++pointer;
                continue;
            }

            double impurityGain = gainByCriteria(labels, pointer, ids);
            double threshold = (features[pointer - 1] + features[pointer]) / (double) 2;

            log.debug("{}/{}/{} -> impurityGain: {}", featureId, threshold, pointer, impurityGain);

            if (impurityGain > bestGain) {
                bestGain = impurityGain;
                bestThreshold = threshold;
                log.debug("Better pair found: {}/{} -> impurityGain: {}", featureId, threshold, impurityGain);
            }

            ++pointer;

            if (loopCounter++ % 99999 == 0){
                log.warn("Warning: In task {} loopCounter {}", featureId, loopCounter);
            }
        }

        return new double[]{bestGain, bestThreshold};
    }

    @Override
    public void initialize(DataSet d) {
        dataSet = d;
        existIds = IntStream.range(0, d.getInstanceLength()).toArray();
        depth = 0;
    }

}
