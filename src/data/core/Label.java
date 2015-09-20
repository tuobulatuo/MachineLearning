package data.core;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public class Label {

    private float[] vector = null;

    private HashSet<String> categories = null;

    private boolean isCategory = false;

//    public Label (float[] data) {
//
//        this.vector = data;
//    }

    public Label(float[] data, HashSet<String> categories) {

        this.vector = data;
        this.categories = categories;
    }

    public boolean isCategoryLabel() {

        return isCategory;
    }

    public double getRow(int rowNum) {

        return vector[rowNum];
    }

    public float[] getVector() {

        return vector;
    }

    public HashSet<String> getCategories() {
        return categories;
    }

    public Label subLableByRow (int[] subIndexes) {
        float[] suVector = new float[subIndexes.length];
        IntStream.range(0, subIndexes.length).forEach(i -> suVector[i] = vector[subIndexes[i]]);
        return new Label(suVector, this.categories);
    }

    public static void main(String[] args) {

    }
}
