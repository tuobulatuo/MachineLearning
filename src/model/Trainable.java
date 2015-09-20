package model;

import data.DataSet;

/**
 * Created by hanxuan on 9/18/15 for machine_learning.
 */
public interface Trainable {
    void train();
    Predictable offer();
    void initialize(DataSet d);
}
