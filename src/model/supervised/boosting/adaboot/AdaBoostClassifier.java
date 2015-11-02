package model.supervised.boosting.adaboot;

import data.DataSet;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public interface AdaBoostClassifier {

    <T> void boostInitialize(DataSet data, T info);

    void boost();

    double boostPredict(double[] feature);

}
