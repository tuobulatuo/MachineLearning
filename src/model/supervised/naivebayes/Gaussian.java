package model.supervised.naivebayes;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.HashSet;

/**
 * Created by hanxuan on 10/16/15 for machine_learning.
 */
public class Gaussian extends NaiveBayes{

    private static final Logger log = LogManager.getLogger(Gaussian.class);

    private static final double STABLE_COEF = 1E-9D;

    private NormalDistribution[][] featureProbabilityDistribution = null;


    @Override
    protected double[] predictClassProbability(double[] features) {

        double[] probabilities = new double[classCount];
        for (int classIndex : indexClassMap.keySet()) {
            for (int i = 0; i < featureLength; i++) {
                probabilities[classIndex] += featureProbabilityDistribution[classIndex][i].logDensity(features[i]);
            }
        }
        return probabilities;
    }

    @Override
    protected void naiveBayesTrain() {

        Arrays.stream(indexClassMap.keySet().toArray()).parallel().forEach(
                x -> {
                    int classIndex = (int) x;
                    for (int i = 0; i < featureLength; i++) {
                        double[] fi = data.getFeatureColFilteredByLabel(i, new HashSet<>(Arrays.asList(classIndex)));

                        double meanI = StatUtils.mean(fi);
                        double sdI = Math.pow(StatUtils.variance(fi), 1) + STABLE_COEF;

                        NormalDistribution nd = new NormalDistribution(meanI, sdI);
                        featureProbabilityDistribution[classIndex][i] = nd;
                    }
                }
        );
    }

    @Override
    protected void naiveBayesInit() {
        featureProbabilityDistribution = new NormalDistribution[classCount][featureLength];
        log.info("Gaussian init finished...");
    }
}
