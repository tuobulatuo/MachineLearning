package data;

import data.core.Label;
import data.core.AMatrix;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.TIntSet;

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

//    private int kFold = 0;

    private int[][] kFoldIndex = null;

    public DataSet(AMatrix m, Label l) {

        this.featureMatrix = m;
        this.labels = l;
    }

    public DataSet(AMatrix m, Label l, int k) {

        this(m, l);
//        this.kFold = k;
        TIntList instancesIndex = new TIntArrayList(IntStream.range(0, featureMatrix.getInstanceLength()).toArray());
        instancesIndex.shuffle(new Random());
        kFoldIndex = new int[k][];
        int pointer = 0;
        for (int i = 0; i < k; i++) {
            int len = Math.min(featureMatrix.getInstanceLength() / k, instancesIndex.size() - pointer);
            int[] a = new int[len];
            for (int j = 0; j < len; j++) {
                a[j] = instancesIndex.get(pointer++);
            }
            kFoldIndex[i] = a;
        }
    }

    public double getEntry(int i, int j){

        return featureMatrix.getEntry(i, j);
    }

    public double[] getInstance(int i) {

        return featureMatrix.getRow(i);
    }

    public double[] getFeature(int i) {

        return featureMatrix.getCol(i);
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

    public String[] getFeatureNames() {

        return featureMatrix.getFeatureNames();
    }

    public Label getLabels() {

        return labels;
    }

}
