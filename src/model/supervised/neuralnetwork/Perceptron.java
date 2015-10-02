package model.supervised.neuralnetwork;

import algorithms.gradient.Decent;
import algorithms.gradient.GradientDecent;
import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/1/15 for machine_learning.
 */
public class Perceptron implements Predictable, Trainable, GradientDecent, Decent {

    public static double ALPHA = 0.001;

    public static final int BUCKET_COUNT = 1;

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

        return hypothesis(feature, w) > 0 ? 1 : -1;
    }

    @Override
    public void train() {

        double[] initTheta = RandomUtils.randomZeroOneArray(data.getFeatureLength());
        double finalCost = loop(data.getInstanceLength(), BUCKET_COUNT, initTheta);
        log.info("Training finished, final cost: {}", finalCost);
        log.info("theta: {}", initTheta);
        log.info("Norm theta: {}", Arrays.stream(initTheta).map(x -> x / initTheta[0]).toArray());
        w = initTheta;
    }

    @Override
    public <T> void gGradient(int start, int end, T params) {

        double[] theta = (double[]) params;

        TIntHashSet mistakes = new TIntHashSet();
        IntStream.range(0, data.getInstanceLength()).forEach(
                i -> {
                    if (hypothesis(data.getInstance(i), theta) * data.getLabel(i) <= 0) mistakes.add(i);
                }
        );

        double[] g = new double[theta.length];
        int[] mistakesArray = mistakes.toArray();
        Arrays.stream(mistakesArray).forEach(
                j -> {
                    double[] X = data.getInstance(j);
                    double y = data.getLabel(j);
                    IntStream.range(0, g.length).forEach(i -> g[i] += X[i] * theta[i] * y);
                }
        );

        IntStream.range(0, theta.length).forEach(i -> theta[i] += ALPHA * g[i]);
    }

    @Override
    public <T> double cost(T params) {

        double[] theta = (double[]) params;

        AtomicDouble cost = new AtomicDouble(0);
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(0, data.getInstanceLength()).parallel().forEach(
                i -> {
                    double[] X = data.getInstance(i);
                    double test = hypothesis(X, theta) * data.getLabel(i);
                    if (test <= 0) {
                        cost.getAndAdd(test * -1.0);
                        counter.getAndIncrement();
                    }
                }
        );

        log.info("ITERATION: {}, TOTAL_MISTAKE : {}, COST: {}", ITER_COUNT++, counter.get(), cost.get());

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
