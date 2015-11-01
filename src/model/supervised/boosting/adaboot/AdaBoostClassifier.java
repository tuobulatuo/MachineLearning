package model.supervised.boosting.adaboot;

import data.DataSet;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public interface AdaBoostClassifier {

    void boostInitialize(DataSet data, double[] weights);

    void boost();

    double boostPredict(double[] feature);

}
