package model.supervised.cart;

/**
 * Created by hanxuan on 9/17/15.
 */
public class RegressionTree extends Tree{

    @Override
    public double growByCriteria(int[] group1, int[] group2) {
        return growByCostDrop(group1, group2);
    }

    private double growByCostDrop(int[] group1, int[] group2) {
        return 0;
    }
}
