package data.core;

import org.neu.util.rand.RandomUtils;

import java.util.HashMap;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public class Label {

    private float[] vector = null;

    private HashMap<Integer, Integer> classIndexMap = null;

    private HashMap<Integer, Integer> indexClassMap = null;

//    private boolean isCategory = false;

    public Label(float[] data, HashMap<Integer, Integer> categories) {

        this.vector = data;
        this.classIndexMap = categories;
        if (classIndexMap != null) {
            indexClassMap = new HashMap<>();
            for (int k : classIndexMap.keySet()) {
                indexClassMap.put(classIndexMap.get(k), k);
            }
        }
    }

    public double getLabelQuotient(int category) {

        float l = (float) (category * 1.0);
        int counter = 0;
        for (float e : vector) if (e == l) ++ counter;
        return counter / (double) vector.length;
    }

//    public boolean isCategoryLabel() {
//
//        return isCategory;
//    }

    public double getRow(int rowNum) {

        return vector[rowNum];
    }

    public float[] getVector() {

        return vector;
    }

    public HashMap<Integer, Integer> getClassIndexMap() {
        return classIndexMap;
    }

    public HashMap<Integer, Integer> getIndexClassMap() {
        return indexClassMap;
    }

    public Label subLabelByRow(int[] subIndexes) {
        float[] suVector = new float[subIndexes.length];
        IntStream.range(0, subIndexes.length).forEach(i -> suVector[i] = vector[subIndexes[i]]);
        return new Label(suVector, this.classIndexMap);
    }

    public Label clone() {
        return subLabelByRow(RandomUtils.getIndexes(vector.length));
    }

    public static void main(String[] args) {

    }
}
