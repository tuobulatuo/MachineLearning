package utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 10/2/15 for machine_learning.
 */
public class NumericalComputation {

    private static Logger log = LogManager.getLogger(NumericalComputation.class);

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidGradient(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    public static double logOdds(double prob) {
        if (prob <= 0.0 || prob >= 1.0) log.warn("logOdds for {}", prob);
        return Math.log(prob)  - Math.log(1.0D - prob);
    }
}
