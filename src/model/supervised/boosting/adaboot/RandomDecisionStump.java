package model.supervised.boosting.adaboot;

import data.DataSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Random;

/**
 * Created by hanxuan on 10/31/15 for machine_learning.
 */
public class RandomDecisionStump extends DecisionStump{

    private static Logger log = LogManager.getLogger(RandomDecisionStump.class);

    public RandomDecisionStump(){}

    public RandomDecisionStump(int depth, DataSet dataSet, int[] existIds, double[] weights){
        super(depth, dataSet, existIds, weights);
        log.info("RandomDecisionStump initialized ...");
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

        left = new RandomDecisionStump(this.depth + 1, this.dataSet, leftGroup, weights);
        right = new RandomDecisionStump(this.depth + 1, this.dataSet, rightGroup, weights);
    }

    protected double weightedError(double[] labels, int position, int[] sortedIds) {
        return new Random().nextDouble();
    }
}
