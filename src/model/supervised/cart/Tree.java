package model.supervised.cart;

import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.Arrays;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntPredicate;

/**
 * Created by hanxuan on 9/17/15.
 */
public abstract class Tree implements Trainable, Predictable{

    private static final Logger log = LogManager.getLogger(Tree.class);

    public static int TREE_ID = 0;

    public static int MAX_DEPTH = 10;

    public static int MIN_INSTANCE_COUNT = 5;

    public static int MAX_THREADS = 7;

    protected int td;

    protected Tree left = null;

    protected Tree right = null;

    protected double treeLabel = 0;

    protected int depth = 0;

    protected int[] existIds = null;

    protected DataSet dataSet = null;

    private int featureId = Integer.MIN_VALUE;

    private double featureThreshold = Integer.MAX_VALUE;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    public Tree(int depth, DataSet dataSet, int[] existIds) {
        this.depth = depth;
        this.dataSet = dataSet;
        this.existIds = existIds;
        td = TREE_ID ++;
    }


    protected abstract double gainByCriteria(int[] ids, int position);

    protected abstract boolean lessThanImpurityGainThreshold(double gain);

    protected abstract void setTreeLabel();

    protected abstract void newTree(int[] leftGroup, int[] rightGroup);

    private boolean stopGrow() {

        return depth >= MAX_DEPTH || existIds.length <= MIN_INSTANCE_COUNT;
    }

    private boolean pure() {
        double first = dataSet.getLabel(existIds[0]);
        IntPredicate pred = (i) -> dataSet.getLabel(i) != first;
        return Arrays.stream(existIds).anyMatch(pred);
    }

    public void grow() {

        if (pure()) {
            log.info("[STOP GROW] pure node ...");
            return;
        }

        if (stopGrow()) {

            log.info("[STOP GROW] depth >= MAX_DEPTH || existIds.length <= MIN_INSTANCE_COUNT");
            setTreeLabel();
            return;
        }

        int featureLength = dataSet.getFeatureLength();
        final AtomicInteger bestFeatureId = new AtomicInteger(Integer.MIN_VALUE);
        final AtomicDouble bestThreshold = new AtomicDouble(Integer.MIN_VALUE);
        final AtomicDouble bestGain = new AtomicDouble(Integer.MIN_VALUE);

        service = Executors.newFixedThreadPool(MAX_THREADS);
        countDownLatch = new CountDownLatch(featureLength);

        for (int i = 0; i < featureLength; i++) {
            final int FEATURE_ID = i;
            service.submit(() -> {

                try {

                    int[] ids = existIds.clone();
                    double[] features = new double[ids.length];
                    fillFeature(ids, features, FEATURE_ID);
                    SortIntDoubleUtils.sort(ids, features);

                    int pointer = 1;
                    while (pointer < ids.length) {
                        if (features[pointer] == features[pointer - 1]) {
                            ++ pointer;
                            continue;
                        }

                        double impurityGain = gainByCriteria(ids, pointer);
                        double threshold = features[pointer];

                        log.info("{}/{}/{} -> impurityGain: {}", FEATURE_ID, threshold, pointer, impurityGain);

                        if (impurityGain > bestGain.get()) {
                            bestGain.set(impurityGain);
                            bestThreshold.set(threshold);
                            bestFeatureId.set(FEATURE_ID);
                            log.info("Better pair found: {}/{} -> impurityGain: {}", FEATURE_ID, threshold, impurityGain);
                        }
                    }
                } catch (Throwable e) {
                    log.error(e.getMessage(), e);
                }
            });
            countDownLatch.countDown();
        }

        try {
            TimeUnit.SECONDS.sleep(1);
            countDownLatch.await();
        } catch (InterruptedException e) {
            log.error(e.getMessage(), e);
        }
        service.shutdown();

        log.info("*KEY STEP: all task finished, service shutdown ...");

        log.info("Best FeatureId: {}, threshold: {}, gain: {}", bestFeatureId.get(), bestThreshold.get(), bestGain.get());

        if (lessThanImpurityGainThreshold(bestGain.get())) {

            log.info("[STOP GROW] lessThanImpurityGainThreshold ...");
            setTreeLabel();
            return;
        }

        featureId = bestFeatureId.get();
        featureThreshold = bestThreshold.get();

        splitGrow();

    }

    private void splitGrow() {

        TIntHashSet left = new TIntHashSet();
        TIntHashSet right = new TIntHashSet();
        for (int i : existIds) {
            if (dataSet.getEntry(i, featureId) < featureThreshold) {
                left.add(i);
            }else {
                right.add(i);
            }
        }
        newTree(left.toArray(), right.toArray());
    }


    private void fillFeature(int[] ids, double[] featureArray, int featureId) {
        for (int i = 0; i < ids.length; i++) {
            featureArray[i] = dataSet.getEntry(ids[i], featureId);
        }
    }

    @Override
    public void predict() {

    }

    @Override
    public void train() {
        this.grow();
    }


}
