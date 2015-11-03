package model.supervised.boosting;

import data.DataSet;
import performance.Evaluator;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public interface Boost {

    void boostConfig(int iteration, String boosterClassName, Evaluator evaluator, DataSet testData)
            throws Exception;

    void printRoundReport();

}
