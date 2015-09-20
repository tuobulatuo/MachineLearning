package data.core;


import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

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

    protected IntPredicate indicator = i -> !booleanColumnIndicator[i];


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

    //************************


    public void shiftCompressNormalize() {

        final double[] ds = new double[featureLength];
        IntStream.range(0, featureLength).filter(indicator).parallel().forEach(i -> ds[i] = colMin(i));
        log.debug("min: {}", ds);
        IntStream.range(0, featureLength).filter(indicator).parallel().forEach(i -> colSubtract(i, ds[i]));
        IntStream.range(0, featureLength).filter(indicator).parallel().forEach(i -> ds[i] = colMax(i));
        log.debug("max: {}", ds);
        IntStream.range(0, featureLength).filter(indicator).parallel().forEach( i -> colMultiply(i, 1.0 / ds[i]));
    }


    public void meanVarianceNormalize() {

        final double[] mean = new double[featureLength];
        final double[] sd = new double[featureLength];
        IntStream.range(0, featureLength).filter(indicator).parallel().forEach(
                i -> {
                    mean[i] = colMean(i);
                    sd[i] = colSd(i);
                }
        );

        log.debug("mean: {}", mean);
        log.debug("sd: {}", sd);

        IntStream.range(0, featureLength).filter(indicator).parallel().forEach(i -> colSubtract(i, mean[i]));
        IntStream.range(0, featureLength).filter(indicator).parallel().forEach( i -> colMultiply(i, 1.0 / sd[i]));
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
