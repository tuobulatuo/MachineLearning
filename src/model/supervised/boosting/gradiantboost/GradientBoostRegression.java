package model.supervised.boosting.gradiantboost;

import data.DataSet;
import data.core.Label;
import model.Predictable;
import model.Trainable;
import model.supervised.boosting.Boost;
import model.supervised.boosting.gradiantboost.gradientboostor.GradientBooster;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.Evaluator;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class GradientBoostRegression implements Predictable, Trainable, Boost{

    public static double OUTLIER_QUANTILE = 95.0;

    public static boolean NEED_REPORT = false;

    private static Logger log = LogManager.getLogger(GradientBoostRegression.class);

    private DataSet trainData = null;

    private DataSet roundData = null;

    protected DataSet testData = null;

    private double[] roundDelta = null;

    private double[] roundTestRss = null;

    private double[] roundTrainRss = null;

    private boolean[] roundIndicator = null;

    private GradientBooster[] boosters = null;

    private Evaluator roundEvaluator = null;

    private Label roundLabels = null;

    private Percentile percentile = new Percentile();

    @Override
    public double predict(double[] feature) {
        return IntStream.range(0, boosters.length).filter(i -> roundIndicator[i]).
                mapToDouble(i -> boosters[i].boostPredict(feature)).sum();
    }

    @Override
    public void train() {

        for (int i = 0; i < boosters.length; i++) {

            boosters[i].boostInitialize(roundData, "");
            boosters[i].boost();
            roundIndicator[i] = true;

            double[] gradient = new double[roundData.getInstanceLength()];
            for (int j = 0; j < gradient.length; j++) {
                gradient[j] = trainData.getLabel(j) - predict(trainData.getInstance(j));
            }

            double delta = percentile.evaluate(Arrays.stream(gradient).map(x -> Math.abs(x)).toArray(), OUTLIER_QUANTILE);

            log.info("round {}, delta {}", i, delta);

            float[] labels = new float[gradient.length];
            for (int j = 0; j < labels.length; j++) {
                if (Math.abs(gradient[j]) <= delta) {
                    labels[j] = (float) gradient[j];
                }else {
                    labels[j] = (float) (delta * (gradient[j] == 0 ? 0 : (gradient[j] > 0 ? 1 : -1)));
                }
            }

            roundData = new DataSet(trainData.getFeatureMatrix(), new Label(labels, null));

            if (NEED_REPORT) {
                statisticReport(i, delta);
            }
        }

        log.info("GradientBoostRegression training finished ...");

        if (NEED_REPORT) {
            printRoundReport();
        }

    }

    private void statisticReport(int round, double delta){

        roundDelta[round] = delta;

        roundEvaluator.initialize(trainData, this);
        roundEvaluator.getPredictLabel();
        roundTrainRss[round] = roundEvaluator.evaluate();

        roundEvaluator.initialize(testData, this);
        roundEvaluator.getPredictLabel();
        roundTestRss[round] = roundEvaluator.evaluate();
    }

    @Override
    public void printRoundReport() {
        log.info("================ round report ================");
        log.info("roundDelta: {}", roundDelta);
        log.info("roundTrainRss: {}", roundTrainRss);
        log.info("roundTestRss: {}", roundTestRss);
        log.info("================ ============ ================");
    }

    @Override
    public void initialize(DataSet d) {
        this.trainData = d;
        roundLabels = trainData.getLabels().clone();
        roundData = new DataSet(trainData.getFeatureMatrix(), roundLabels);
    }

    @Override
    public void boostConfig(int iteration, String boosterClassName, Evaluator evaluator, DataSet testData)
            throws Exception{

        GradientBooster[] boosters = new GradientBooster[iteration];
        for (int i = 0; i < iteration; i++) {
            boosters[i] = (GradientBooster) Class.forName(boosterClassName).getConstructor().newInstance();
        }
        this.boosters = boosters;

        roundDelta = new double[boosters.length];
        roundTestRss = new double[boosters.length];
        roundTrainRss = new double[boosters.length];
        roundIndicator = new boolean[boosters.length];

        roundEvaluator = evaluator;
        this.testData = testData;

        log.info("GradientBoostRegression configTrainable: ");
        log.info("boosters count: {}", boosters.length);
        log.info("boosters CLASS: {}", boosterClassName);
    }
}
