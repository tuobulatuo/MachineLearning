package model.supervised.linearmodel;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 9/17/15.
 */
public class NormalEquation implements Predictable, Trainable{

    private static final Logger log = LogManager.getLogger(NormalEquation.class);

    public static double LAMBDA = 0;

    private RealMatrix w = null;    // m by 1

    private RealMatrix matrix = null;   // n by m

    private RealMatrix identity = null;

    private RealMatrix y = null;    // n by 1

    public NormalEquation() {}

    @Override
    public double predict(double[] feature) {

        double[][] feature1MatrixData = new double[1][];
        feature1MatrixData[0] = feature;
        RealMatrix feature1Matrix = new Array2DRowRealMatrix(feature1MatrixData, false);
        return feature1Matrix.multiply(w).getEntry(0, 0) > 0.5 ? 1 : 0;
    }

    @Override
    public double score(double[] feature) {
        double[][] feature1MatrixData = new double[1][];
        feature1MatrixData[0] = feature;
        RealMatrix feature1Matrix = new Array2DRowRealMatrix(feature1MatrixData, false);
        return feature1Matrix.multiply(w).getEntry(0, 0);
    }

    @Override
    public void train() {
        RealMatrix inverseMatrix = matrix.transpose().multiply(matrix);
        if (identity != null) {
            inverseMatrix = inverseMatrix.add(identity);
        }
        RealMatrix pInverse = new LUDecomposition(inverseMatrix).getSolver().getInverse();
        w = pInverse.multiply(matrix.transpose()).multiply(y);

        log.debug("Normal Equation Training finished ...");
    }

    @Override
    public Predictable offer() {
        return this;
    }

    @Override
    public void initialize(DataSet d) {

        int n = d.getInstanceLength();
        double[][] data = new double[n][];
        double[][] label = new double[n][1];
        for (int i = 0; i < n ; i++) {
            data[i] = d.getInstance(i);
            label[i][0] = d.getLabel(i);
        }
        matrix = new Array2DRowRealMatrix(data, false);
        y = new Array2DRowRealMatrix(label, false);

        if (LAMBDA > 0) {
            identity = MatrixUtils.createRealIdentityMatrix(matrix.getColumnDimension());
        }

        log.info("Matrix initialized, dim: {} by {}", matrix.getRowDimension(), matrix.getColumnDimension());
    }
}


