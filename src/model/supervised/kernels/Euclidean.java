package model.supervised.kernels;

import utils.array.ArrayUtil;

/**
 * Created by hanxuan on 12/10/15 for machine_learning.
 */
public class Euclidean implements Kernel{
    @Override
    public double similarity(double[] x1, double[] x2) {
        return - ArrayUtil.euclidean(x1, x2);
    }
}
