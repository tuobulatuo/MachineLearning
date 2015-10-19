package model.supervised.naivebayes;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.NumericalComputation;

import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/15/15 for machine_learning.
 */
public class Multinoulli extends NaiveBayes{

    public static double[] QUOTIENTS = {50};

    public static int BINS = 2;

    private static final Logger log = LogManager.getLogger(Multinoulli.class);

    private double[][] featureThresholds = null;

    private double[][][] featureProbabilityDistribution = null;


    @Override
    protected double[] predictClassProbability(double[] features) {

        double[] probabilities = new double[classCount];
        for (int classIndex : indexClassMap.keySet()) {
            for (int i = 0; i < featureLength; i++) {
                double[] p = featureProbabilityDistribution[classIndex][i];
                int binIndex = binIndex(i, features[i]);
                probabilities[classIndex] += Math.log(p[binIndex]);
            }
        }
        return probabilities;
    }

    @Override
    protected void naiveBayesTrain() {

        Percentile percentile = new Percentile();
        for (int i = 0; i < featureLength; i++) {
            double[] fi = data.getFeatureCol(i);
            log.debug("FI {}", fi);
            for (int j = 0; j < QUOTIENTS.length; j++) {
                featureThresholds[i][j] = percentile.evaluate(fi, QUOTIENTS[j]);
            }
        }

        log.debug("featureThresholds {}", featureThresholds);
        log.debug("indexClassMap {} {}", indexClassMap, indexClassMap.keySet().toArray());

        Arrays.stream(indexClassMap.keySet().toArray()).parallel().forEach(

                x -> {
                    int classIndex = (int) x;
                    for (int i = 0; i < featureLength; i++) {
                        double[] fi = data.getFeatureColFilteredByLabel(i, new HashSet<>(Arrays.asList(classIndex)));
                        Arrays.sort(fi); // small -> large
                        double[] bins = featureProbabilityDistribution[classIndex][i];
                        int counter = 0;
                        int thresholdPointer = 0;
                        for (int j = 0; j < fi.length; j++) {
                            if (thresholdPointer < BINS - 1 && fi[j] > featureThresholds[i][thresholdPointer]) {
                                bins[thresholdPointer] = (counter + 1) / (double) (fi.length + BINS); // laplace smoothing
                                counter = 0;
                                ++thresholdPointer;
                            }
                            ++counter;
                        }
                        bins[thresholdPointer] = (counter + 1) / (double) (fi.length + BINS);
                    }
                }
        );

        log.info("Multinoulli Model training finished ...");
    }

    private int binIndex(int featureIndex, double feature) {

        double[] thresholds = featureThresholds[featureIndex];
        for (int i = 0; i < thresholds.length; ++i) {
            if (feature <= thresholds[i]) return i;
        }
        return thresholds.length;
    }

    @Override
    protected void naiveBayesInit() {

        featureThresholds = new double[featureLength][BINS - 1];
        featureProbabilityDistribution = new double[classCount][featureLength][BINS];

        if (BINS - 1 != QUOTIENTS.length) {
            throw new IllegalArgumentException("BINS - 1 != QUOTIENTS.length");
        }
        log.info("Multinoulli Model initialized, BINS : {}, QUOTIENTS: {}", BINS, QUOTIENTS);
    }
}
