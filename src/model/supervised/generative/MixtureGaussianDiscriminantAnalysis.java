package model.supervised.generative;

import algorithms.parameterestimate.DataPointSet;
import algorithms.parameterestimate.MixtureGaussianEM;
import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/17/15 for machine_learning.
 */
public class MixtureGaussianDiscriminantAnalysis implements Trainable, Predictable{

    public static int COMPONENTS = 3;

    public static int MAX_THREADS = 1;

    private static final Logger log = LogManager.getLogger(MixtureGaussianDiscriminantAnalysis.class);

    private DataSet data = null;

    private HashMap<Integer, Integer> indexClassMap = null;

    private MultivariateNormalDistribution[][] models = null;

    private double[][] componentsPi = null;

    private double[] priors = null;

    private int classCount = Integer.MIN_VALUE;

    private int featureLength = Integer.MIN_VALUE;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    @Override
    public double predict(double[] feature) {

        double[] probabilities = new double[classCount];
        for (int i = 0; i < models.length; i++) {
            probabilities[i] = mixtureDensity(feature, i) * priors[i];
        }
        int[] index = RandomUtils.getIndexes(classCount);
        SortIntDoubleUtils.sort(index, probabilities);
        log.debug(probabilities[probabilities.length - 1]);
        return index[index.length - 1];
    }

    @Override
    public double score(double[] feature) {
        double[] probabilities = new double[classCount];
        for (int i = 0; i < models.length; i++) {
            probabilities[i] = mixtureDensity(feature, i) * priors[i];
        }
        double score = probabilities[1] - probabilities[0];
        return score;
    }


    @Override
    public void train() {

        priors = new double[classCount];
        for (int classIndex : indexClassMap.keySet()) {
            priors[classIndex] = data.getCategoryProportion(classIndex);
        }

        service = Executors.newFixedThreadPool(MAX_THREADS);
        countDownLatch = new CountDownLatch(classCount);
        log.info("Task Count: {}", countDownLatch.getCount());

        for (int classIndex : indexClassMap.keySet()) {
            final int CLASS_INDEX = classIndex;
            service.submit(() -> {

                try {

                    DataPointSet dataPointSet = new DataPointSet(data, CLASS_INDEX, RandomUtils.getIndexes(featureLength));
                    MixtureGaussianEM em = new MixtureGaussianEM(dataPointSet, COMPONENTS);
                    em.run();

                    double[][][] sigma = em.getSigma();
                    double[][] mu = em.getMu();
                    double[] pi = em.getPi();

                    MultivariateNormalDistribution[] mixtureDistribution = new MultivariateNormalDistribution[COMPONENTS];
                    for (int i = 0; i < COMPONENTS; i++) {
                        mixtureDistribution[i] = new MultivariateNormalDistribution(mu[i], sigma[i]);
                    }

                    componentsPi[CLASS_INDEX] = pi;
                    models[CLASS_INDEX] = mixtureDistribution;

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
    public void initialize(DataSet d) {

        this.data =d;
        indexClassMap = d.getLabels().getIndexClassMap();
        classCount = indexClassMap.size();
        featureLength = data.getFeatureLength();
        models = new MultivariateNormalDistribution[classCount][];
        componentsPi = new double[classCount][];
    }

    private double mixtureDensity(double[] feature, int modelIndex) {

        return IntStream.range(0, COMPONENTS).mapToDouble(i ->
                models[modelIndex][i].density(feature) * componentsPi[modelIndex][i]).sum();
    }
}
