package data;

import com.google.common.primitives.Ints;
import data.core.Label;
import data.core.AMatrix;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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
        Label subLabels = labels.subLableByRow(rowIndexes);
        return new DataSet(subMatrix, subLabels);
    }

    public void shiftCompressNorm(float[] min, float[] max) {
        featureMatrix.shiftCompressNormalize(min, max);
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

    public double getLabel(int i) {

        return labels.getRow(i);
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

    public float[] getMeanOrMin() {
        return featureMatrix.getMeanOrMin();
    }

    public float[] getSdOrMax() {
        return featureMatrix.getSdOrMax();
    }

}
