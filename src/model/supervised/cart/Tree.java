package model.supervised.cart;

import data.DataSet;
import gnu.trove.set.TIntSet;
import model.supervised.SModel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 9/17/15.
 */
public abstract class Tree extends SModel{

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

    public void grow() {

    }

    protected abstract double growByCriteria(int[] group1, int[] group2);

}
