package data;

import data.core.Label;
import data.core.AMatrix;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.Set;

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

    public DataSet subDataSetByRow(int[] rowIndexes) {
        AMatrix subMatrix = featureMatrix.subMatrixByRow(rowIndexes);
        Label subLabels = labels.subLabelByRow(rowIndexes);
        return new DataSet(subMatrix, subLabels);
    }

    public void shiftCompressNorm() {
        featureMatrix.shiftCompressNormalize();
    }

    public void meanVarianceNorm(float[] mean, float[] sd) {
        featureMatrix.meanVarianceNormalize(mean, sd);
    }

    public void meanVarianceNorm() {
        featureMatrix.meanVarianceNormalize();
    }

    public double getEntry(int i, int j){

        return featureMatrix.getEntry(i, j);
    }

    public double[] getInstance(int i) {

        return featureMatrix.getRow(i);
    }

    public double[] getFeatureCol(int featureIndex) {
        return featureMatrix.getCol(featureIndex);
    }

    public double[] getFeatureColFilteredByLabel(int col, Set<Integer> ls) {

        TDoubleArrayList filterFeature = new TDoubleArrayList(getInstanceLength());
        double[] feature = featureMatrix.getCol(col);
        for (int i = 0; i < feature.length; i++) {
            if (ls.contains((int) getLabel(i))) filterFeature.add(feature[i]);
        }
        return filterFeature.toArray();
    }

    public double getLabel(int i) {

        return labels.getRow(i);
    }

    public double getCategoryProportion(int category) {
        return labels.getLabelQuotient(category);
    }

    public AMatrix getFeatureMatrix() {

        return featureMatrix;
    }

    public int getInstanceLength() {

        return featureMatrix.getInstanceLength();
    }

    public int getFeatureLength() {

        return featureMatrix.getFeatureLength();
    }

    public Label getLabels() {

        return labels;
    }
}
