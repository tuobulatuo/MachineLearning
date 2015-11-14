package model.supervised.boosting.adaboot.adaboostclassifier;

import data.DataSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public class DecisionStump extends WeightedClassificationTree{

    private static Logger log = LogManager.getLogger(DecisionStump.class);

    public DecisionStump(){
        INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        MAX_DEPTH = 1;
    }

    public DecisionStump(int depth, DataSet dataSet, int[] existIds, double[] weights){
        super(depth, dataSet, existIds, weights);
        log.debug("DecisionStump initialized, MAX_DEPTH restrict to {}, INFORMATION_GAIN_THRESHOLD restrict to {}",
                MAX_DEPTH, INFORMATION_GAIN_THRESHOLD);
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

        left = new DecisionStump(this.depth + 1, this.dataSet, leftGroup, weights);
        right = new DecisionStump(this.depth + 1, this.dataSet, rightGroup, weights);
    }

    @Override
    public double gainByCriteria(double[] labels, int position, int[] sortedIds) {
        return Math.abs(0.5 - weightedError(labels, position, sortedIds));
    }

    protected double weightedError(double[] labels, int position, int[] sortedIds) {

        double error = 0;
        for (int i = 0; i < labels.length; i++) {
            if (i < position) {
                error += labels[i] == 0.0D ? 0 : weights[sortedIds[i]];
            } else {
                error += labels[i] == 1.0D ? 0 : weights[sortedIds[i]];
            }
        }
        return error;
    }

    @Override
    public int bestFeatureId() {
        return featureId;
    }
}
