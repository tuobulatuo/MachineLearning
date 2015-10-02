package model.supervised.linearmodel;

import algorithms.gradient.Decent;
import algorithms.gradient.DecentType;
import algorithms.gradient.GradientDecent;
import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.NumericalComputation;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/30/15 for machine_learning.
 */
public class LogisticGradientDecent implements Predictable, Trainable, Decent, GradientDecent {

    private static final Logger log = LogManager.getLogger(LogisticGradientDecent.class);

    private static double POSITIVE_THRESHOLD = 0.5;

    public static double ALPHA = 0.01;   // learning rate

    public static double LAMBDA = 0.0;  // punish rate

    public static int BUCKET_COUNT = 1;   // mini batch

    protected DecentType type = DecentType.GRADIENT;

    protected DataSet data = null;

    protected double[] w = null;

    public LogisticGradientDecent() {}


    @Override
    public void train() {

        double[] initTheta = new double[data.getFeatureLength()];
        double finalCost = loop(data.getInstanceLength(), BUCKET_COUNT, initTheta);
        log.info("Training finished, final cost: {}", finalCost);
        w = initTheta;
    }

    @Override
    public Predictable offer() {
        return this;
    }

    @Override
    public void initialize(DataSet d) {
        this.data = d;
    }

    @Override
    public double predict(double[] feature) {

        return hypothesis(w, feature) > POSITIVE_THRESHOLD ? 1 : 0;
    }

    @Override
    public <T> void gGradient(int start, int end, T params) {

        double[] theta = (double[]) params;

        double[] g = new double[theta.length];
        IntStream.range(0, g.length).forEach(
                i -> IntStream.range(start, end).parallel().forEach(
                        j -> {
                            double[] X = data.getInstance(j);
                            g[i] += (hypothesis(X, theta) - data.getLabel(j)) * X[i];
                        }
                )
        );
        log.debug("theta: {}",theta);
        log.debug("g : {}", g);
        IntStream.range(1, g.length).forEach(i -> g[i] += LAMBDA * theta[i]);
        IntStream.range(0, theta.length).forEach(i -> theta[i] -= ALPHA * g[i]);
    }

    @Override
    public <T> double cost(T params) {

        double[] theta = (double[]) params;
        int instanceLength = data.getInstanceLength();
        AtomicDouble cost = new AtomicDouble(0);
        IntStream.range(0, instanceLength).parallel().forEach(
                i -> {
                    double y = data.getLabel(i);
                    double hx = hypothesis(data.getInstance(i), theta);
                    cost.getAndAdd(y * Math.log(hx) + (1 - y) * Math.log(1 - hx));
                }
        );

        cost.getAndSet(- cost.doubleValue());

        double punish = LAMBDA * Arrays.stream(theta).map(x -> Math.pow(x, 2)).sum() / 2;

        cost.getAndAdd(punish);

        return cost.doubleValue();
    }

    @Override
    public <T> void parameterGradient(int start, int end, T theta) {
        if (type == DecentType.GRADIENT) {
            gGradient(start, end, theta);
        }else {

        }
    }

    private double hypothesis(double[] x, double[] theta) {
        return NumericalComputation.sigmoid(IntStream.range(0, x.length).mapToDouble(i -> x[i] * theta[i]).sum());
    }
}
