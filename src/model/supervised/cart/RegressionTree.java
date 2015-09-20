package model.supervised.cart;

import data.DataSet;

/**
 * Created by hanxuan on 9/17/15.
 */
public class RegressionTree extends Tree{

    public static final double COST_DROP_THRESHOLD = 1;

    public RegressionTree(int depth, DataSet dataSet, int[] existInstanceIndex) {
        super(depth, dataSet, existInstanceIndex);
    }

    @Override
    public double gainByCriteria(double[] labels, int position) {
        return growByCostDrop(labels, position);
    }

    @Override
    protected boolean lessThanImpurityGainThreshold(double gain) {
        return gain < COST_DROP_THRESHOLD;
    }

    @Override
    protected void setTreeLabel() {

    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

    }

    private double growByCostDrop(double[] labels, int position) {
        return 0;
    }
}
