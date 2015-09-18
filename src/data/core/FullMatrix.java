package data.core;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public class FullMatrix extends AMatrix {

    private static final Logger log = LogManager.getLogger(FullMatrix.class);

    private float[][] data = null;

    public FullMatrix(float[][] data, boolean[] booleansIndicator) {

        this.data = data;
        this.booleanColumnIndicator = booleansIndicator;
        this.instanceLength = data.length;
        this.featureLength = data[0].length;
    }

    public FullMatrix(float[][] data, boolean[] booleansIndicator, String[] featureNames) {
        this(data, booleansIndicator);
        this.featureNames = featureNames;
    }

    @Override
    public double[] getRow(int rowNum) {
        double[] out = new double[featureLength];
        IntStream.range(0, featureLength).parallel().forEach(i -> out[i] = (double) data[rowNum][i]);
        return out;
    }

    @Override
    public double[] getCol(int colNum) {
        double[] out = new double[instanceLength];
        IntStream.range(0, instanceLength).parallel().forEach(i -> out[i] = (double) data[i][colNum]);
        return out;
    }

    @Override
    public double colMean(int colNum) {
        return IntStream.range(0, instanceLength).parallel().mapToDouble(i -> data[i][colNum]).sum() / instanceLength;
    }

    @Override
    public double colSd(int colNum) {

        double mean = colMean(colNum);
        double accumulator = IntStream.range(0, instanceLength).
                mapToDouble(i -> Math.pow(data[i][colNum] - mean, 2)).sum();
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
        IntStream.range(0, instanceLength).forEach(i -> data[i][colNum] -= x);
    }

    @Override
    public void colMultiply(int colNum, double x) {
        IntStream.range(0, instanceLength).forEach(i -> data[i][colNum] *= x);
    }


    public static void main(String[] args) {

        float[][] m0 = new float[][] {
                {1, 96,    26,    26,    55},
                {1, 55,    82,    62,    92},
                {1, 14,    25,    48,    29},
                {1, 15,    93,    36,    76},
                {1, 26,    35,    84,    76},
                {1, 85,    20,    59,    39}

        };
        boolean[] indicator = new boolean[] {true, false, false, false, false, false, false};

        FullMatrix fm = new FullMatrix(m0, indicator);

        log.info("before: {}", m0);

        log.info(fm.colMean(0));

        fm.shiftCompressNormalize();
        IntStream.range(0, fm.instanceLength).forEach(i -> log.info(Arrays.toString(fm.getRow(i))));
        fm.meanVarianceNormalize();
        log.info("after: {}", m0);
    }
}
