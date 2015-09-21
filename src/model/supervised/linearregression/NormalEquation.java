package model.supervised.linearregression;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import model.Predictable;
import model.Trainable;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;

/**
 * Created by hanxuan on 9/17/15.
 */
public class NormalEquation implements Predictable, Trainable{

    private static final Logger log = LogManager.getLogger(NormalEquation.class);

    private RealMatrix w = null;    // m by 1

    private RealMatrix matrix = null;   // n by m

    private RealMatrix y = null;    // n by 1

    public NormalEquation() {}

    public NormalEquation(DataSet dataSet) {
        initialize(dataSet);
    }

    @Override
    public double predict(double[] feature) {

        double[][] feature1MatrixData = new double[1][];
        feature1MatrixData[0] = feature;
        RealMatrix feature1Matrix = new Array2DRowRealMatrix(feature1MatrixData, false);
        return feature1Matrix.multiply(w).getEntry(0, 0);
    }

    @Override
    public void train() {
        RealMatrix inverseMatrix = matrix.transpose().multiply(matrix);
        RealMatrix pInverse = new LUDecomposition(inverseMatrix).getSolver().getInverse();
        w = pInverse.multiply(matrix.transpose()).multiply(y);
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
    }

    public static void main(String[] args) throws IOException {


    }
}


