package model.supervised.svm;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 11/25/15 for machine_learning.
 */
public class SVMsSMO implements Trainable, Predictable{

    private static Logger log = LogManager.getLogger(SVMsSMO.class);

    @Override
    public double predict(double[] feature) {
        return 0;
    }

    @Override
    public void train() {

    }

    @Override
    public void initialize(DataSet d) {

    }
}
