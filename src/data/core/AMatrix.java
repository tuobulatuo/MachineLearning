package data.core;


import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 9/17/15.
 */
public abstract class AMatrix {

    private static Logger log = LogManager.getLogger(AMatrix.class);

    private int instanceLength;

    private int featureLenght;

    private String[] featureNames;

    private boolean[] booleanColumnIndicator;

    private float[] range;

    private float[] mean;

    private float[] sd;

    private float[] median;

    public abstract float[] getRow(int rowNum);

    public abstract float[] getCol(int colNum);

    public abstract float colMean(int colNum);

    public abstract float colSd(int colNum);

    public abstract void colSubtract(int colNum, float mu);

    public abstract void colMultiply(int colNum, float sd);

    public void shiftCompressNormalize() {
//        for (int i)
    }

    public void meanVarianceNormalize() {

    }

    public static void main(String[] args) {

        System.out.println("before log4j");
        log.info("hi {} {}", 1, 2);
        log.debug("debug");
        log.warn("warn");
        log.error("error");

        System.out.println();

        System.out.print("after log4j");
    }

}
