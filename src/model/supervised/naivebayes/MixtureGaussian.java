package model.supervised.naivebayes;

import algorithms.parameterestimate.DataPointSet;
import algorithms.parameterestimate.MixtureGaussianEM;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/18/15 for machine_learning.
 */
public class MixtureGaussian extends NaiveBayes {

    public static int COMPONENTS = 3;

    public static int MAX_THREADS = 1;

    private static final Logger log = LogManager.getLogger(MixtureGaussian.class);

    private MultivariateNormalDistribution[][][] models = null;

    private double[][][] componentsPi = null;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    @Override
    protected double[] predictClassProbability(double[] features) {
        double[] probabilities = new double[classCount];
        for (int i = 0; i < models.length; i++) {
            probabilities[i] = mixtureDensity(features, i);
        }
        return probabilities;
    }

    @Override
    protected void naiveBayesTrain() {

        service = Executors.newFixedThreadPool(MAX_THREADS);
        countDownLatch = new CountDownLatch(classCount);
        log.info("Task Count: {}", countDownLatch.getCount());

        for (int classIndex : indexClassMap.keySet()) {
            final int CLASS_INDEX = classIndex;
            service.submit(() -> {

                try {

                    for (int featureIndex = 0; featureIndex < featureLength; featureIndex++) {

                        DataPointSet dataPointSet = new DataPointSet(data, CLASS_INDEX, new int[]{featureIndex});
                        MixtureGaussianEM em = new MixtureGaussianEM(dataPointSet, COMPONENTS);
                        em.run();

                        double[][][] sigma = em.getSigma();
                        double[][] mu = em.getMu();
                        double[] pi = em.getPi();

                        MultivariateNormalDistribution[] mixtureDistribution = new MultivariateNormalDistribution[COMPONENTS];
                        for (int i = 0; i < COMPONENTS; i++) {
                            mixtureDistribution[i] = new MultivariateNormalDistribution(mu[i], sigma[i]);
                        }

                        componentsPi[CLASS_INDEX][featureIndex] = pi;
                        models[CLASS_INDEX][featureIndex] = mixtureDistribution;

                        log.info("class: {} feature {} finished EM ...", CLASS_INDEX, featureIndex);
                    }

                    log.info("class: {} finished EM ...", CLASS_INDEX);

                }catch (Throwable t) {
                    log.error(t.getMessage(), t);
                }

                countDownLatch.countDown();

            });
        }

        try {
            TimeUnit.MILLISECONDS.sleep(10);
            countDownLatch.await();
        }catch (Throwable t) {
            log.error(t.getMessage(), t);
        }
        service.shutdown();

        log.info("MixtureGaussianDiscriminantAnalysis Training finished, service shutdown ...");
    }

    @Override
    protected void naiveBayesInit() {
        log.info("{} {} {}", classCount, featureLength, COMPONENTS);
        models = new MultivariateNormalDistribution[classCount][featureLength][COMPONENTS];
        componentsPi = new double[classCount][featureLength][COMPONENTS];
    }

    private double mixtureDensity(double[] feature, int modelIndex) {

        return
        IntStream.range(0, featureLength).mapToDouble(featureIndex ->
                        Math.log(
                                IntStream.range(0, COMPONENTS).mapToDouble(
                                        component ->
                                                models[modelIndex][featureIndex][component].density(
                                                        new double[]{feature[featureIndex]}) *
                                                        componentsPi[modelIndex][featureIndex][component]).sum()
                        )
        ).sum();
    }
}
