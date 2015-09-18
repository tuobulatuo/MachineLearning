package data;

import data.core.Label;
import data.core.AMatrix;

/**
 * Created by hanxuan on 9/17/15.
 */

public class DataSet {

    private AMatrix featureMatrix = null;

    private Label labels = null;

    public DataSet(AMatrix m, Label l) {
        this.featureMatrix = m;
        this.labels = l;
    }

    public AMatrix getFeatureMatrix() {

        return featureMatrix;
    }

//    public void setFeatureMatrix(AMatrix featureMatrix) {
//        this.featureMatrix = featureMatrix;
//    }

    public Label getLabels() {

        return labels;
    }

//    public void setLabels(Label labels) {
//        this.labels = labels;
//    }

}
