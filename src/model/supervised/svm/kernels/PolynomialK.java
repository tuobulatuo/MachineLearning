package model.supervised.svm.kernels;

import utils.array.ArrayUtil;

/**
 * Created by hanxuan on 11/25/15 for machine_learning.
 */
public class PolynomialK implements Kernel {

    public static double DEGREE = 2.0D;

    public static double COEF = 1.0D;

    public static double GAMMA = 1.0D;

    @Override
    public double similarity(double[] x1, double[] x2) {
        double innerProduct = ArrayUtil.innerProduct(x1, x2);
        return Math.pow((GAMMA * innerProduct) + COEF, DEGREE);
    }
}


