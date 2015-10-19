package model.supervised.generative;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/15/15 for machine_learning.
 */
public class GaussianDiscriminantAnalysis implements Predictable, Trainable{

    public static boolean COV_DISTINCT = false;

    private static Logger log = LogManager.getLogger(GaussianDiscriminantAnalysis.class);

    private static double STABLE_COEF = 1E-5D;

    private MultivariateNormalDistribution[] models = null;

    private double[] priors = null;

    private int classCount = Integer.MIN_VALUE;

    private int featureLength = Integer.MIN_VALUE;

    private double[][] covariancesCommon = null;

    private DataSet data = null;

    private HashMap<Integer, Integer> indexClassMap = null;

    @Override
    public double predict(double[] feature) {
        double[] probabilities = new double[classCount];
        for (int i = 0; i < models.length; i++) {
            probabilities[i] = models[i].density(feature) * priors[i];
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
            probabilities[i] = models[i].density(feature) * priors[i];
        }
        double score = Math.log(probabilities[1]) - Math.log(probabilities[0]);
        return score;
    }

    @Override
    public void train() {

        priors = new double[classCount];
        for (int category : indexClassMap.keySet()) {
            priors[category] = data.getCategoryProportion(category);
        }

        for (int category : indexClassMap.keySet()) {
            final double[] means = new double[featureLength];
            fillMean(new HashSet<>(Arrays.asList(category)), means);
            MultivariateNormalDistribution mnd;
            if (COV_DISTINCT) {
                final double[][] covariances = new double[featureLength][featureLength];
                fillCovariance(new HashSet<>(Arrays.asList(category)), covariances);
                mnd = new MultivariateNormalDistribution(means, covariances);
            } else {
                mnd = new MultivariateNormalDistribution(means, covariancesCommon);
            }
            models[category] = mnd;
        }

        log.info("GDA Model training finished ... ");
    }

    private void fillMean(Set<Integer> category, double[] initMeans) {

        for (int i = 0; i < featureLength; i++) {
            double[] x = data.getFeatureColFilteredByLabel(i, category);
            log.debug("label: {} x: {}", category, Arrays.toString(x));
            initMeans[i] = Arrays.stream(x).average().getAsDouble();
        }
    }

    private void fillCovariance(Set<Integer> category, double[][] initCovs) {

        Covariance covariance = new Covariance();
        for (int i = 0; i < featureLength; i++) {
            double[] x1 = data.getFeatureColFilteredByLabel(i, category);
            final int FEATURE_I = i;
            IntStream.range(i, featureLength).parallel().forEach(
                    j -> {
                        double[] x2 = data.getFeatureColFilteredByLabel(j, category);
                        double cov = covariance.covariance(x1, x2, true);

                        if (cov == 0) {
                            cov = STABLE_COEF;
                        }

                        initCovs[FEATURE_I][j] = cov;
                        initCovs[j][FEATURE_I] = cov;
                    }
            );
        }
    }

    @Override
    public void initialize(DataSet d) {

        data = d;
        indexClassMap = d.getLabels().getIndexClassMap();
        classCount = indexClassMap.size();
        featureLength = data.getFeatureLength();
        models = new MultivariateNormalDistribution[classCount];
        if (!COV_DISTINCT) {
            covariancesCommon = new double[data.getFeatureLength()][data.getFeatureLength()];
            fillCovariance(indexClassMap.keySet(), covariancesCommon);
        }
    }
}
