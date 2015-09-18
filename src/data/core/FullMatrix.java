package data.core;

/**
 * Created by hanxuan on 9/17/15.
 */
public class FullMatrix extends AMatrix {

    private float[][] data;

    @Override
    public float[] getRow(int rowNum) {
        return new float[0];
    }

    @Override
    public float[] getCol(int colNum) {
        return new float[0];
    }

    @Override
    public float colMean(int colNum) {
        return 0;
    }

    @Override
    public float colSd(int colNum) {
        return 0;
    }

    @Override
    public void colSubtract(int colNum, float mu) {

    }

    @Override
    public void colMultiply(int colNum, float sd) {

    }
}
