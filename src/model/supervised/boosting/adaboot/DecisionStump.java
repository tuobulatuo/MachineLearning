package model.supervised.boosting.adaboot;

import data.DataSet;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import model.supervised.cart.ClassificationTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.array.ArraySumUtil;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortDoubleDoubleUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public class DecisionStump extends WeightedClassificationTree{

    private static Logger log = LogManager.getLogger(DecisionStump.class);

    public DecisionStump(){}

    public DecisionStump(int depth, DataSet dataSet, int[] existIds, double[] weights){
        super(depth, dataSet, existIds, weights);
        log.info("DecisionStumpTest initialized ...");
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
}
