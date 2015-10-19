package model.supervised.naivebayes;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.HashMap;

/**
 * Created by hanxuan on 10/15/15 for machine_learning.
 */
public abstract class NaiveBayes implements Predictable, Trainable{

    private static final Logger log = LogManager.getLogger(NaiveBayes.class);

    protected DataSet data = null;

    protected int featureLength = Integer.MIN_VALUE;

    protected HashMap<Integer, Integer> indexClassMap = null;

    protected int classCount = Integer.MIN_VALUE;

    private double[] priors = null;


    //******************************//

    protected abstract double[] predictClassProbability(double[] features);

    protected abstract void naiveBayesTrain();

    protected abstract void naiveBayesInit();

    //******************************//

    @Override
    public double predict(double[] feature) {

        double[] probabilities = predictClassProbability(feature);
        for (int i = 0; i < classCount; i++) {
            probabilities[i] += Math.log(priors[i]);
        }
        int[] indexes = RandomUtils.getIndexes(classCount);
        SortIntDoubleUtils.sort(indexes, probabilities);
        return indexes[indexes.length - 1];
    }

    @Override
    public double score(double[] feature) {
        double[] probabilities = predictClassProbability(feature);
        double score = probabilities[1] - probabilities[0];
        return score;
    }

    @Override
    public void train() {

        for (int category : indexClassMap.keySet()) {
            priors[category] = data.getCategoryProportion(category);
        }

        naiveBayesTrain();
    }

    @Override
    public void initialize(DataSet d) {

        data = d;
        featureLength = d.getFeatureLength();
        indexClassMap = d.getLabels().getIndexClassMap();
        classCount = indexClassMap.size();
        priors = new double[classCount];
        naiveBayesInit();

        log.info("Naive Bayes Model initializing, classCount: {}", classCount);
    }
}
