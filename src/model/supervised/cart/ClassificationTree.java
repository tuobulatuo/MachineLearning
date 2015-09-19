package model.supervised.cart;

/**
 * Created by hanxuan on 9/17/15.
 */
public class ClassificationTree extends Tree{

    @Override
    public double growByCriteria(int[] group1, int[] group2) {
        return growByInformationGain(group1, group2);
    }

    private double growByInformationGain(int[] group1, int[] group2) {
        return 0;
    }
}
