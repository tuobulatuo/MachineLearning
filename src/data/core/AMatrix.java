package data.core;


import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.function.IntPredicate;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public abstract class AMatrix {

    private static Logger log = LogManager.getLogger(AMatrix.class);

    protected int instanceLength = 0;

    protected int featureLength = 0;

    protected String[] featureNames = null;

    protected boolean[] booleanColumnIndicator = null;

    protected IntPredicate nonBooleanFeature = i -> !booleanColumnIndicator[i];

    private float[] meanOrMin = null;

    private float[] sdOrMax = null;

    //************************

    public abstract double getEntry(int rowNun, int column);

    public abstract double[] getRow(int rowNum);

    public abstract double[] getCol(int colNum);

    public abstract double colMean(int colNum);

    public abstract double colSd(int colNum);

    public abstract double colMin(int colNum);

    public abstract double colMax(int colNum);

    public abstract void colSubtract(int colNum, double x);

    public abstract void colMultiply(int colNum, double x);

    public abstract AMatrix subMatrixByRow(int[] rowIndexes);

    //************************


    public void shiftCompressNormalize() {

        meanOrMin = new float[featureLength];
        sdOrMax = new float[featureLength];
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach(i -> meanOrMin[i] = (float) colMin(i));
        log.debug("min: {}", meanOrMin);
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach(i -> sdOrMax[i] = (float) (colMax(i) - colMin(i)));
        log.debug("max: {}", sdOrMax);

        scn();
    }

    public void shiftCompressNormalize(float[] min, float[] max) {
        meanOrMin = min;
        sdOrMax = max;
        scn();
    }

    private void scn() {

        filterZeroSd();
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach(i -> colSubtract(i, meanOrMin[i]));
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach(i -> colMultiply(i, 1.0 / sdOrMax[i]));
    }


    public void meanVarianceNormalize() {

        meanOrMin = new float[featureLength];
        sdOrMax = new float[featureLength];
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach(
                i -> {
                    meanOrMin[i] = (float) colMean(i);
                    sdOrMax[i] = (float) colSd(i);
                }
        );

        log.debug("meanOrMin: {}", meanOrMin);
        log.debug("sdOrMax: {}", sdOrMax);
        mvn();
    }

    public void meanVarianceNormalize(float[] mean, float[] sd) {
        meanOrMin = mean;
        sdOrMax = sd;
        mvn();
    }

    private void mvn() {

        filterZeroSd();
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach(i -> colSubtract(i, meanOrMin[i]));
        IntStream.range(0, featureLength).filter(nonBooleanFeature).parallel().forEach( i -> colMultiply(i, 1.0 / sdOrMax[i]));
    }

    private void filterZeroSd() {

        IntStream.range(0, sdOrMax.length).filter(nonBooleanFeature).forEach(i -> {
            if (sdOrMax[i] == 0) {
                sdOrMax[i] = Integer.MAX_VALUE;
                log.warn("WARNING: Feature {}: sdOrMax == 0, set it to  Integer.MAX_VALUE !!", i);
            }
        });
    }


    public int getFeatureLength() {
        return featureLength;
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public int getInstanceLength() {
        return instanceLength;
    }

    public boolean[] getBooleanColumnIndicator() {
        return booleanColumnIndicator;
    }

    public static void main(String[] args) {

        Predicate<String> predicate1 = (s) -> s.length() > 0;
        Predicate<String> predicate2 = predicate1.and((s) -> s.length() < 100);

        log.info(predicate2.test("123"));

        System.out.println("before log4j");
        log.info("hi {} {}", 1, 2);
        log.debug("debug");
        log.warn("warn");
        log.error("error");

        System.out.println();
        System.out.print("after log4j");
    }

}
