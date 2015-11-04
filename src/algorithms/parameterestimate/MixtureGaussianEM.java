package algorithms.parameterestimate;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;
import utils.random.RandomUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/17/15 for machine_learning.
 */
public class MixtureGaussianEM implements EM {

    public static int MAX_ROUND = 500;

    public static int PRINT_GAP = 100;

    public static double THRESHOLD = 1E-5;

    private static double STABLE_COEF = 1E-1;

    private int round = 0;

    private static Logger log = LogManager.getLogger(MixtureGaussianEM.class);

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

    public MixtureGaussianEM(DataPointSet data, int components) {

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

        for (int component = 0; component < components; component++) {
            pi[component] = 1 / (double) components;
            mu[component] = RandomUtils.randomSumOneArray(featureLength);
        }

        for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
            z[dataIndex] =  RandomUtils.randomSumOneArray(components);
        }

        sigma();

        log.debug("init sigma: {}", sigma);

        log.info("MixtureGaussianEM initialized: [{}] components, [{}] data points, [{}] feature", components, dataLength, featureLength);
    }

    @Override
    public void e() {

        z();
    }

    @Override
    public void m() {

        pi();
        mu();
        sigma();

        if (round++ % PRINT_GAP == 0) {
            log.info("[{}] round likelihood {}", round, currentLikelihood);
        }
    }

    @Override
    public boolean convergence() {

        if ((round > MAX_ROUND) || (Math.abs(currentLikelihood - previousLikelihood) < THRESHOLD)) {
            log.info("***** MixtureGaussianEM convergence met *****");
            log.info("({}) / ({})", round, MAX_ROUND);
            log.info("{} - {} < {}", currentLikelihood, previousLikelihood, THRESHOLD);
            log.info("Final Params: ");
            log.info("mu {}:", mu);
            log.info("sigma {}", sigma);
            log.info("pi {}", pi);
            log.info("**************** Now stop EM ****************");
            return true;
        }

        return false;
    }

    private void pi() {
        for (int component = 0; component < components; component++) {
            double zm = zm(component);
            pi[component] = zm / dataLength;
        }

        log.debug("sum Pi : {}", ArraySumUtil.sum(pi));
    }

    private void mu () {

        IntStream.range(0, components).forEach(i -> Arrays.fill(mu[i], 0));

        for (int component = 0; component < components; component++) {
            double zm = zm(component);
            for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
                double[] x = data.getI(dataIndex);
                for (int featureIndex = 0; featureIndex < featureLength; featureIndex++) {
                    mu[component][featureIndex] += x[featureIndex] * z[dataIndex][component] / zm;
                }
            }

            log.debug("mu M ({}), {}:", component, mu[component]);
        }
    }

    private void sigma() {

        IntStream.range(0, components).forEach(
                i -> IntStream.range(0, featureLength).forEach(j -> Arrays.fill(sigma[i][j], 0)));

        for (int component = 0; component < components; component++) {
            double zm = zm(component);
            double[][] sigmam = sigma[component];
            for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
                double[] x = data.getI(dataIndex);
                double[] mum = mu[component];
                for (int featureIndex1 = 0; featureIndex1 < featureLength; featureIndex1++) {

                    sigmam[featureIndex1][featureIndex1] += Math.pow(x[featureIndex1] - mum[featureIndex1], 2) * z[dataIndex][component] / zm;

                    for (int featureIndex2 = featureIndex1 + 1; featureIndex2 < featureLength; featureIndex2++) {
                        double accu = (x[featureIndex1] - mum[featureIndex1]) * (x[featureIndex2] - mum[featureIndex2]);
                        accu = accu * z[dataIndex][component] / zm;
                        sigmam[featureIndex1][featureIndex2] += accu;
                        sigmam[featureIndex2][featureIndex1] += accu;
                    }
                }
            }

            for (int i = 0; i < featureLength; i++) {
                sigmam[i][i] += STABLE_COEF;    // smoothing diagonal (Math: put a normal prior to mu)
            }

            log.debug("sigma M ({}), {}:", component, sigmam);
        }
    }

    private void z() {

        IntStream.range(0, dataLength).forEach(i -> Arrays.fill(z[i], 0));

        double[] densitySum = new double[dataLength];
        for (int component = 0; component < components; component++) {

            MultivariateNormalDistribution distribution = null;
            try{
                distribution = new MultivariateNormalDistribution(mu[component], sigma[component]);
            }catch (Exception e){
                log.error(e.getMessage(), e);
                log.error("sigmam: {}", sigma[component]);
                System.exit(-1);
            }

            for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
                double[] x = data.getI(dataIndex);
                double density = distribution.density(x);

                if (density == 0) {
                    log.debug("0 density for data {} component {}", dataIndex, component);
                }

                z[dataIndex][component] = density * pi[component];

                densitySum[dataIndex] += density * pi[component];
            }
        }

        double likelihood = 0;
        for (int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
            double[] zi = z[dataIndex];
            double normalizeFactor = densitySum[dataIndex];
            if (normalizeFactor == 0) {
                Arrays.fill(zi, 1 / (double) components);
            }else {
                likelihood += Math.log(normalizeFactor);
                IntStream.range(0, components).forEach(i -> zi[i] /= normalizeFactor);
            }
            log.debug("zi {} {}", dataIndex, zi);
        }

        double s = 0;
        for (int i = 0; i < components; i++) {
            s += zm(i);
        }

        log.debug("z sum {}", s);

        likelihood /= dataLength;
        previousLikelihood = currentLikelihood;
        currentLikelihood = likelihood;

    }

    private double zm (int component) {
        double re = 0;
        for (int i = 0; i < dataLength; i++) re += z[i][component];
        return re;
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
