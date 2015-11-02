package data.core;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public class SparseMatrix extends AMatrix{

    private static final Logger log = LogManager.getLogger(SparseMatrix.class);

    private int[][] index = null;

    private float[][] data = null;

    private float[] missingFeatureDefault = null;

    public SparseMatrix(int[][] index, float[][] data, int featureLength, boolean[] booleansIndicator) {
        this.index = index;
        this.data = data;
        this.featureLength = featureLength;
        this.instanceLength = data.length;
        missingFeatureDefault = new float[featureLength];
        this.booleanColumnIndicator = booleansIndicator;
    }

    public SparseMatrix(int[][] index, float[][] data, int featureLength, boolean[] booleansIndicator, float[] missingFeatureDefault) {
        this(index, data, featureLength, booleansIndicator);
        this.missingFeatureDefault = missingFeatureDefault;
    }

    @Override
    public double getEntry(int rowNun, int column) {
        int colIndex = Arrays.binarySearch(index[rowNun], column);
        return colIndex >= 0 ? data[rowNun][colIndex] : missingFeatureDefault[column];
    }

    @Override
    public double[] getRow(int rowNum) {
        double[] row = new double[featureLength];
        IntStream.range(0, featureLength).forEach(i -> row[i] = missingFeatureDefault[i]);
        IntStream.range(0, index[rowNum].length).forEach(i -> row[index[rowNum][i]] = data[rowNum][i]);
        return row;
    }

    @Override
    public double[] getCol(int colNum) {
        double[] col = new double[data.length];
        IntStream.range(0, col.length).forEach(i -> col[i] = getEntry(i, colNum));
        return col;
    }

    @Override
    public double colMean(int colNum) {
        return Arrays.stream(getCol(colNum)).average().getAsDouble();
    }

    @Override
    public double colSd(int colNum) {

        double mean = colMean(colNum);
        double accumulator = IntStream.range(0, instanceLength).
                mapToDouble(i -> Math.pow(getEntry(i, colNum) - mean, 2)).sum();
        double sd = Math.sqrt((accumulator / (instanceLength - 1)));
        return sd;
    }

    @Override
    public double colMin(int colNum) {
        return Arrays.stream(getCol(colNum)).min().getAsDouble();
    }

    @Override
    public double colMax(int colNum) {
        return Arrays.stream(getCol(colNum)).max().getAsDouble();
    }

    @Override
    public void colSubtract(int colNum, double x) {
        missingFeatureDefault[colNum] -= x;
        for (int i = 0; i < data.length; i++) {
            int colIndex = Arrays.binarySearch(index[i], colNum);
            if (colIndex >= 0){
                data[i][colIndex] -= x;
            }
        }
    }

    @Override
    public void colMultiply(int colNum, double x) {
        missingFeatureDefault[colNum] *= x;
        for (int i = 0; i < data.length; i++) {
            int colIndex = Arrays.binarySearch(index[i], colNum);
            if (colIndex >= 0){
                data[i][colIndex] *= x;
            }
        }
    }

    @Override
    public AMatrix subMatrixByRow(int[] rowIndexes) {

        int[][] subIndexes = new int[rowIndexes.length][];
        float[][] subData = new float[rowIndexes.length][];
        IntStream.range(0, rowIndexes.length).forEach(i -> subData[i] = data[rowIndexes[i]].clone());
        IntStream.range(0, rowIndexes.length).forEach(i -> subIndexes[i] = index[rowIndexes[i]].clone());
        log.debug("subMatrixByRow finished ...");
        return new SparseMatrix(subIndexes,subData, featureLength, booleanColumnIndicator.clone(), missingFeatureDefault.clone());
    }
}
