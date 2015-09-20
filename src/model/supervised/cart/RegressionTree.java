package model.supervised.cart;

import data.DataSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;

/**
 * Created by hanxuan on 9/17/15.
 */
public class RegressionTree extends Tree{

    private static final Logger log = LogManager.getLogger(RegressionTree.class);

    public static double COST_DROP_THRESHOLD = 1;

    private double squareError = Integer.MAX_VALUE;

    public RegressionTree() {}

    public RegressionTree(int depth, DataSet dataSet, int[] existIds) {

        super(depth, dataSet, existIds);

        double[] labels = Arrays.stream(existIds).mapToDouble(id -> dataSet.getLabel(id)).toArray();
        squareError = cost(labels);

        log.info("Tree {} @depth {} constructed, squareError: {}, has {} points ...", td, depth, squareError, existIds.length);
    }

    @Override
    public double gainByCriteria(double[] labels, int position) {
        return CostDrop(labels, position);
    }

    @Override
    protected boolean lessThanImpurityGainThreshold(double gain) {
        return gain < COST_DROP_THRESHOLD;
    }

    @Override
    protected void setTreeLabel() {

        treeLabel = Arrays.stream(existIds).mapToDouble(id -> dataSet.getLabel(id)).average().getAsDouble();
        log.info("[LEAF NODE] id: {}, label: {}", td, treeLabel);
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

        left = new RegressionTree(depth + 1, dataSet, leftGroup);
        right = new RegressionTree(depth + 1, dataSet, rightGroup);
    }

    private double CostDrop(double[] labels, int position) {

        double[] left = Arrays.copyOfRange(labels, 0, position);
        double[] right = Arrays.copyOfRange(labels, position, labels.length);
        return squareError - cost(left) - cost(right);
    }

    private double cost (double[] a) {
        double mean = Arrays.stream(a).average().getAsDouble();
        return Arrays.stream(a).map(x -> Math.pow(x - mean, 2)).sum();
    }

    public static void main(String[] args) {
        RegressionTree rt = new RegressionTree();
        double[] a = {1,2,3};
        log.info(rt.cost(a));
    }
}
