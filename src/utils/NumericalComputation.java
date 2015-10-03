package utils;

/**
 * Created by hanxuan on 10/2/15 for machine_learning.
 */
public class NumericalComputation {

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidGradient(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }
}
