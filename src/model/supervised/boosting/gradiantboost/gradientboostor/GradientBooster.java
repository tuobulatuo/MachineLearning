package model.supervised.boosting.gradiantboost.gradientboostor;

import data.DataSet;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public interface GradientBooster {

    <T> void boostInitialize(DataSet data, T info);

    void boost();

    double boostPredict(double[] feature);
}
