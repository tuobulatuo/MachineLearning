package model.supervised.bagging;

import data.DataSet;
import performance.Evaluator;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public interface Bagging {
    void baggingConfig(int iteration, String trainableClassName, Evaluator evaluator, DataSet testData);
}
