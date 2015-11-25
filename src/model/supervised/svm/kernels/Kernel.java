package model.supervised.svm.kernels;

/**
 * Created by hanxuan on 11/25/15 for machine_learning.
 */
public interface Kernel {
    double similarity(double[] x1, double[] x2);
}
