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

    public Label getLabels() {

        return labels;
    }

}
