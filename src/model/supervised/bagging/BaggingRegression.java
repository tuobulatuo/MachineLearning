package model.supervised.bagging;

import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class BaggingRegression extends BootstrapAveraging{

    @Override
    public double predict(double[] feature) {
        double ans = IntStream.range(0, predictables.length).mapToDouble(
                i -> predictables[i].predict(feature)).average().getAsDouble();
        return ans;
    }
}
