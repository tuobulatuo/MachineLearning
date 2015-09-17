package data;

import data.core.Label;
import data.core.Matrix;

/**
 * Created by hanxuan on 9/17/15.
 */

public class DataSet {

    private Matrix featureMatrix = null;
    private Label labels = null;

    public Matrix getFeatureMatrix() {
        return featureMatrix;
    }

    public void setFeatureMatrix(Matrix featureMatrix) {
        this.featureMatrix = featureMatrix;
    }

    public Label getLabels() {
        return labels;
    }

    public void setLabels(Label labels) {
        this.labels = labels;
    }



}
