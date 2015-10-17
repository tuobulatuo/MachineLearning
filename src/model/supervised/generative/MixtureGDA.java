package model.supervised.generative;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;

/**
 * Created by hanxuan on 10/17/15 for machine_learning.
 */
public class MixtureGDA implements Trainable, Predictable{

    private static final Logger log = LogManager.getLogger(MixtureGDA.class);

    private DataSet data = null;

    private HashMap<Integer, Integer> indexClassMap = null;


    @Override
    public double predict(double[] feature) {
        return 0;
    }

    @Override
    public void train() {

    }

    @Override
    public void initialize(DataSet d) {

        this.data =d;
        indexClassMap = d.getLabels().getIndexClassMap();
    }
}
