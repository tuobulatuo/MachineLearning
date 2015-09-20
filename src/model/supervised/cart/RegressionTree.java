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
    public double gainByCriteria(int[] ids, int position) {
        return growByCostDrop(ids, position);
    }

    @Override
    protected boolean lessThanImpurityGainThreshold(double gain) {
        return gain < COST_DROP_THRESHOLD;
    }

    @Override
    protected void setTreeLabel() {

    }

    @Override
    protected void newTree(int[] leftGroup, int[] rightGroup) {

    }

    private double growByCostDrop(int[] ids, int position) {
        return 0;
    }
}
