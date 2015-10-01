package model;

import data.DataSet;

/**
 * Created by hanxuan on 9/18/15 for machine_learning.
 */
public interface Trainable {
    void train();

    default Predictable offer(){
        return (Predictable) this;
    }

    void initialize(DataSet d);
}
