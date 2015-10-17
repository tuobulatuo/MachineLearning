package algorithms.parameterestimate;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/17/15 for machine_learning.
 */
public class MixGaussianEM implements EM {

    public static int MAX_ROUND = 5000;

    public static int PRINT_GAP = 100;

    public static double THRESHOLD = 1E-5;

    private static double STABLE_COEF = 1E-10;

    private int round = 0;

    private static Logger log = LogManager.getLogger(MixGaussianEM.class);

    private DataPointSet data = null;

    private int components = Integer.MIN_VALUE;

    private int dataLength = Integer.MIN_VALUE;

    private int featureLength = Integer.MIN_VALUE;

    private double[][] z = null;

    private double[][][] sigma = null;

    private double[][] mu = null;

    private double[] pi = null;

    private double previousLikelihood = Integer.MIN_VALUE;

    private double currentLikelihood = Integer.MAX_VALUE;

    public MixGaussianEM (DataPointSet data, int components) {

        this.data = data;
        this.components = components;
        dataLength = data.size();
        featureLength = data.width();
    }

    @Override
    public void initialize() {

        z = new double[dataLength][components];
        sigma = new double[components][featureLength][featureLength];
        mu = new double[components][featureLength];
        pi = new double[components];

        IntStream.range(0, components).forEach(i -> Arrays.fill(z[i], 1 / (double) components));
        pi();
        mu();
        sigma();

        log.info("MixGaussianEM initialized: {} components, {} data points, {} feature", components, dataLength, featureLength);
    }

    @Override
    public void e() {
        z();

        if (round++ % PRINT_GAP == 0) {
            log.info("[{}] round likelihood {}", currentLikelihood);
        }
    }

    @Override
    public void m() {
        pi();
        mu();
        sigma();
    }

    @Override
    public boolean convergence() {
        return (round > MAX_ROUND) || (currentLikelihood - previousLikelihood < THRESHOLD);
    }

    private void pi() {
        for (int component = 0; component < components; component++) {
            double zm = zm(component);
            pi[component] = zm / dataLength;
        }
    }

    private void mu () {
        for (int component = 0; component < components; component++) {
            double zm = zm(component);
            for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
                double[] x = data.getI(dataIndex);
                for (int featureIndex = 0; featureIndex < featureLength; featureIndex++) {
                    mu[component][featureIndex] += x[featureIndex] * z[dataIndex][component] / zm;
                }
            }
        }
    }

    private void sigma() {

        for (int component = 0; component < components; component++) {
            double zm = zm(component);
            double[][] sigmam = sigma[component];
            for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
                double[] x = data.getI(dataIndex);
                double[] mum = mu[component];
                for (int featureIndex1 = 0; featureIndex1 < featureLength; featureIndex1++) {
                    for (int featureIndex2 = featureIndex1; featureIndex2 < featureLength; featureIndex2++) {
                        double accu = (x[featureIndex1] - mum[featureIndex1]) * (x[featureIndex2] - mum[featureIndex2]);
                        accu = accu * z[dataIndex][component] / zm;
                        sigmam[featureIndex1][featureIndex2] += accu;
                        sigmam[featureIndex2][featureIndex1] += accu;
                    }
                }
            }

            for (int i = 0; i < featureLength; i++) {
                sigmam[i][i] += STABLE_COEF;    // smoothing diagonal
            }
        }
    }

    private void z() {

        double likelihood = 0;

        double[] densitySum = new double[dataLength];
        for (int component = 0; component < components; component++) {
            MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(mu[component], sigma[component]);
            for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
                double[] x = data.getI(dataIndex);
                double density = distribution.density(x);
                z[dataIndex][component] = density * pi[component];
                densitySum[dataIndex] += density * pi[component];

                likelihood += Math.log(density * pi[component]);
            }
        }

        for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
            double[] zi = z[dataIndex];
            double normalizeFactor = densitySum[dataIndex];
            Arrays.setAll(zi, i -> zi[i] / normalizeFactor );
        }

        previousLikelihood = currentLikelihood;
        currentLikelihood = likelihood;
    }

    private double zm (int component) {
        double re = 0;
        for (int i = 0; i < dataLength; i++) re += z[i][component];
        return re;
    }

    public double[][] getZ() {
        return z;
    }

    public double[][][] getSigma() {
        return sigma;
    }

    public double[][] getMu() {
        return mu;
    }

    public double[] getPi() {
        return pi;
    }
}
