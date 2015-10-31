package model.supervised.boosting.adaboot;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public interface AdaBoostClassifier {


    void boost(double[] weights);

    double boostPredict(double[] feature);
}
