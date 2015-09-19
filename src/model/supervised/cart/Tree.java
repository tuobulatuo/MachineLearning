package model.supervised.cart;

import data.DataSet;
import gnu.trove.set.TIntSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 9/17/15.
 */
public abstract class Tree implements Trainable, Predictable{

    private static final Logger log = LogManager.getLogger(Tree.class);

    protected Tree left = null;

    protected Tree right = null;

    protected TIntSet existInstanceIndex = null;

    protected DataSet dataSet = null;

    public Tree() {

    }

    @Override
    public void predict() {

    }

    @Override
    public void train() {

    }

    protected abstract double growByCriteria(int[] group1, int[] group2);

    public void grow() {

        int featureLength = dataSet.getFeatureLength();
        int instanceLenght = dataSet.getInstanceLength();



    }



}
