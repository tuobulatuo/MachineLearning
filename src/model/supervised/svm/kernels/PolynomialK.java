package model.supervised.svm.kernels;

import utils.array.ArrayUtil;

/**
 * Created by hanxuan on 11/25/15 for machine_learning.
 */
public class PolynomialK implements Kernel {

    public static double ORDER = 2.0D;

    public static double CONSTANT = 1.0D;

    @Override
    public double similarity(double[] x1, double[] x2) {
        double innerProduct = ArrayUtil.innerProduct(x1, x2);
        return Math.pow((innerProduct + CONSTANT), ORDER);
    }
}


