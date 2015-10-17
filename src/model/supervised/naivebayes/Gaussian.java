package model.supervised.naivebayes;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.array.ArraySumUtil;
import utils.NumericalComputation;

import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;

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
                double p = featureProbabilityDistribution[classIndex][i].density(features[i]);
                probabilities[classIndex] += Math.log(p);
            }
        }
        IntStream.range(0, classCount).forEach(i -> probabilities[i] = NumericalComputation.sigmoid(probabilities[i]));
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
