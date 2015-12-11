package model.supervised.perceptron;

import algorithms.gradient.Decent;
import algorithms.gradient.GradientDecent;
import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/1/15 for machine_learning.
 */
public class Perceptron implements Predictable, Trainable, GradientDecent, Decent {

    public static double ALPHA = 0.001;

    public static int BUCKET_COUNT = 1;

    public static double COST_DECENT_THRESHOLD = 0.00000001;

    public static int MAX_ROUND = 5000;

    public static int PRINT_GAP = 500;

    private static int ITER_COUNT = 1;

    private static final Logger log = LogManager.getLogger(Perceptron.class);

    private double[] w = null;

    private DataSet data = null;

    public Perceptron() {}

    @Override
    public void initialize(DataSet d) {
        this.data = d;
    }

    @Override
    public double predict(double[] feature) {
        return score(feature) > 0 ? 1 : -1;
    }

    @Override
    public double score(double[] feature) {
        return hypothesis(feature, w);
    }

    @Override
    public void train() {
        double[] initTheta = new double[data.getFeatureLength()];
        double finalCost = loop(data.getInstanceLength(), BUCKET_COUNT, initTheta, COST_DECENT_THRESHOLD, MAX_ROUND, PRINT_GAP);
        log.info("Training finished, final cost: {}", finalCost);
        log.info("theta: {}", initTheta);
        log.info("Norm theta: {}", Arrays.stream(initTheta).map(x -> x / initTheta[0]).toArray());
        w = initTheta;
    }

    @Override
    public <T> void gGradient(int start, int end, T params) {

        double[] theta = (double[]) params;

        IntStream.range(0, data.getInstanceLength()).forEach(
                i -> {
                    double[] X = data.getInstance(i);
                    double y = data.getLabel(i);
                    if (hypothesis(X, theta) * y <= 0) {
                        IntStream.range(0, theta.length).forEach(j -> theta[j] += X[j] * y);
                    }
                }
        );
    }

    @Override
    public <T> double cost(T params) {

        double[] theta = (double[]) params;

        AtomicDouble cost = new AtomicDouble(0);
        AtomicInteger mistakeCounter = new AtomicInteger(0);
        IntStream.range(0, data.getInstanceLength()).forEach(
                i -> {
                    double[] X = data.getInstance(i);
                    double test = hypothesis(X, theta) * data.getLabel(i);
                    if (test <= 0) {
                        cost.getAndAdd(test * -1.0);
                        mistakeCounter.getAndIncrement();
                    }
                }
        );

        log.info("ITERATION: {}, TOTAL_MISTAKE : {}, COST: {}", ITER_COUNT++, mistakeCounter.get(), cost.get());

        return cost.get();
    }

    @Override
    public <T> void parameterGradient(int start, int end, T theta) {
        gGradient(start, end, theta);
    }

    private double hypothesis(double[] X, double[] theta) {
        return IntStream.range(0, X.length).mapToDouble(i -> X[i] * theta[i]).sum();
    }
}
