package model.semisupervised.activelearning;

import data.DataSet;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import model.supervised.boosting.adaboot.SAMME;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import utils.sort.SortIntDoubleUtils;

import java.util.Random;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class ActiveAdaBoost implements Predictable, Trainable, ActiveLearning {

    public static double PERCENT_THRESHOLD = 0.5;

    public static double PERCENT_START = 0.05;

    public static double PERCENT_ADD = 0.02;

    private static Logger log = LogManager.getLogger(ActiveAdaBoost.class);

    private DataSet fullData = null;

    private DataSet testSet = null;

    private SAMME samme = null;

    private TIntHashSet labelledIds = null;

    private TIntHashSet unlabelledIds = null;

    private int round = 0;

    private TDoubleArrayList percentReport;

    private TDoubleArrayList accuracyReport;

    public ActiveAdaBoost(int iteration, String AdaBoostClassifierClassName, ClassificationEvaluator evaluator, DataSet testSet) {
        try {
            samme = new SAMME();
            samme.boostConfig(iteration, AdaBoostClassifierClassName, evaluator, testSet);
            this.testSet = testSet;
        }catch (Exception e) {
            log.info(e.getMessage(), e);
        }
    }

    @Override
    public double predict(double[] feature) {
        return samme.predict(feature);
    }

    @Override
    public void train() {
        loop();
        log.info("ActiveAdaBoost training finished ...");
        log.info("==================== Report ====================");
        log.info("accuracy: {}", accuracyReport.toArray());
        log.info("percent: {}", percentReport.toArray());
        log.info("==================== ====== ====================");
    }

    @Override
    public void initialize(DataSet d) {

        fullData = d;
        int instanceLength = d.getInstanceLength();
        Random rand = new Random();
        labelledIds = new TIntHashSet((int) (instanceLength * PERCENT_START));
        unlabelledIds = new TIntHashSet(instanceLength);
        for (int i = 0; i < instanceLength; i++) {
            if (rand.nextDouble() < PERCENT_START) {
                labelledIds.add(i);
            }else {
                unlabelledIds.add(i);
            }
        }

        accuracyReport = new TDoubleArrayList();
        percentReport = new TDoubleArrayList();
    }

    @Override
    public boolean converge() {
        return labelledIds.size() / (double) fullData.getInstanceLength() > PERCENT_THRESHOLD;
    }

    @Override
    public void activeTrain() {
        DataSet data = fullData.subDataSetByRow(labelledIds.toArray());
        samme.initialize(data);
        samme.train();
    }

    @Override
    public void addData() {

        testSetEvaluation();

        double[] ambiguity = new double[unlabelledIds.size()];
        int[] ids = unlabelledIds.toArray();
        for (int i = 0; i < ids.length; i++) {
            ambiguity[i] = Math.abs(samme.score(fullData.getInstance(ids[i])));
        }
        SortIntDoubleUtils.sort(ids, ambiguity);

        for (int i = 0; i < ids.length * PERCENT_ADD; i++) {
            unlabelledIds.remove(ids[i]);
            labelledIds.add(ids[i]);
        }
    }

    private void testSetEvaluation() {

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, this);
        evaluator.getPredictLabel();
        double accuracy = evaluator.evaluate();
        double percent = labelledIds.size() / (double) fullData.getInstanceLength();
        percentReport.add(percent);
        accuracyReport.add(accuracy);

        log.info("round: {}", round++);
        log.info("percent: {}", percent);
        log.info("TEST accuracy: {}", accuracy);
    }
}
