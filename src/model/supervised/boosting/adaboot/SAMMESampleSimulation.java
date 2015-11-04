package model.supervised.boosting.adaboot;

import model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassifier;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;
import utils.random.RandomUtils;

import java.util.concurrent.TimeUnit;

/**
 * Created by hanxuan on 11/1/15 for machine_learning.
 */
public class SAMMESampleSimulation extends SAMME {

    public static int SAMPLE_SIZE_COEF = 1;

    private static Logger log = LogManager.getLogger(SAMMESampleSimulation.class);

    private int[] indexes = null;

    @Override
    public void train() {

        indexes = RandomUtils.getIndexes(weights.length);

        for (int i = 0; i < adaBoostClassifiers.length; i++) {

            AdaBoostClassifier classifier = adaBoostClassifiers[i];

            if (i == 0) {
                classifier.boostInitialize(trainingData, indexes);
            }else {
                EnumeratedIntegerDistribution integerDistribution = new EnumeratedIntegerDistribution(indexes, weights);
                int[] trainingIndexes = integerDistribution.sample(indexes.length * SAMPLE_SIZE_COEF);
                classifier.boostInitialize(trainingData, trainingIndexes);
            }

            classifier.boost();
            double error = getWeightedError(classifier);
            roundError[i] = error;

            alpha[i] = Math.log((1 - error) / error) / 2 + Math.log(classCount - 1);

            modifyWeights(classifier, alpha[i]);
            ArraySumUtil.normalize(weights);

            if (NEED_ROUND_REPORT) {
                statisticReport(i, error);
            }

            log.info("{} round boosting finished ...", i);
        }

        log.info("SAMMESampleSimulation Training finished ...");

        if (NEED_ROUND_REPORT) {
            printRoundReport();
        }

        try{
            TimeUnit.SECONDS.sleep(1);
        }catch (Exception e){
            log.error(e.getMessage(), e);
        }
    }
}
