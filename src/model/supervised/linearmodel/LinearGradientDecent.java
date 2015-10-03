package model.supervised.linearmodel;

import algorithms.gradient.Decent;
import algorithms.gradient.DecentType;
import algorithms.gradient.GradientDecent;
import algorithms.gradient.NewtonDecent;
import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/30/15 for machine_learning.
 */
public class LinearGradientDecent implements Predictable, Trainable, Decent, GradientDecent , NewtonDecent{

    public static double ALPHA = 0.01;   // learning rate

    public static double LAMBDA = 0.0;  // punish rate

    public static int BUCKET_COUNT = 1;   // mini batch

    private static final Logger log = LogManager.getLogger(LinearGradientDecent.class);

    protected DecentType type = DecentType.GRADIENT;

    protected DataSet data = null;

    protected double[] w = null;

    public LinearGradientDecent() {}

    @Override
    public double predict(double[] feature) {
        return hypothesis(feature, w);
    }

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
    public <T> void gGradient(int start, int end, T params) {

        double[] theta = (double[]) params;

        double[] g = new double[theta.length];
        IntStream.range(0, g.length).forEach(
                i -> IntStream.range(start, end).parallel().forEach(
                        j -> {
                            double[] X = data.getInstance(j);
                            g[i] += (hypothesis(X, theta) - data.getLabel(j)) * X[i] / (end - start);
                        }
                )
        );
        log.debug("theta: {}",theta);
        log.debug("g : {}", g);
        IntStream.range(1, g.length).forEach(i -> g[i] += LAMBDA * theta[i] / (end - start));
        IntStream.range(0, theta.length).forEach(i -> theta[i] -= ALPHA * g[i]);

    }

    @Override
    public <T> double cost(T params) {

        double[] theta = (double[]) params;

        int instanceLength = data.getInstanceLength();
        AtomicDouble cost = new AtomicDouble(0);
        IntStream.range(0, instanceLength).parallel().forEach(
            i -> cost.getAndAdd(
                    Math.pow((data.getLabel(i) - hypothesis(data.getInstance(i), theta)), 2)
            )
        );

        cost.getAndSet(cost.doubleValue() / (double) instanceLength / 2.0);

        double punish = LAMBDA * Arrays.stream(theta).map(x -> Math.pow(x, 2)).sum() / 2 / instanceLength;

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
        return IntStream.range(0, x.length).mapToDouble(i -> x[i] * theta[i]).sum();
    }

    public static void main(String[] args) {
        LinearGradientDecent lm = new LinearGradientDecent();
        System.out.println(lm.hypothesis(new double[]{1, 2}, new double[]{3, 4}));
    }

    @Override
    public double[] nGradient(DataSet data, int start, int end, double[] theta) {
        return new double[0];
    }
}
