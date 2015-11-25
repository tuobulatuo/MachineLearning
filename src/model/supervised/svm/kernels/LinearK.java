package model.supervised.svm.kernels;

import utils.array.ArrayUtil;

/**
 * Created by hanxuan on 11/25/15 for machine_learning.
 */
public class LinearK implements Kernel {

    @Override
    public double similarity(double[] x1, double[] x2) {
        return ArrayUtil.innerProduct(x1, x2);
    }
}
