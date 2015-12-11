package model.supervised.perceptron;

import algorithms.gradient.Decent;
import algorithms.gradient.GradientDecent;
import data.DataSet;
import model.Predictable;
import model.Trainable;
import model.supervised.kernels.Kernel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


/**
 * Created by hanxuan on 12/11/15 for machine_learning.
 */
public class DualPerceptron implements Trainable, Predictable, GradientDecent, Decent{


    private static Logger log = LogManager.getLogger(DualPerceptron.class);

    public static int BUCKET_COUNT = 1;

    public static double COST_DECENT_THRESHOLD = 0.00000001;

    public static int MAX_ROUND = 5000;

    public static int PRINT_GAP = 500;

    private static int ITER_COUNT = 1;

    private double[] m = null;

    private DataSet data = null;

    private int instanceLength = -1;

    private Kernel kernel = null;

    public DualPerceptron (String kernelClassName) throws Exception{
        kernel = (Kernel) Class.forName(kernelClassName).newInstance();
        log.info("DualPerceptron use kernel {}", kernelClassName);
    }

    @Override
    public double predict(double[] feature) {
        return hypothesis(feature) > 0 ? 1 : -1;
    }

    @Override
    public void train() {

        loop(instanceLength, BUCKET_COUNT, m, COST_DECENT_THRESHOLD, MAX_ROUND, PRINT_GAP);
        log.info("DualPerceptron training finished ...");
    }

    @Override
    public void initialize(DataSet d) {
        this.data = d;
        instanceLength = data.getInstanceLength();
        m = new double[instanceLength];
    }

    @Override
    public <T> double cost(T params) {

        double cost = 0;
        int misTakeCount = 0;
        for (int i = 0; i < instanceLength; i++) {
            double test = hypothesis(data.getInstance(i)) * data.getLabel(i);
            if (test <= 0) {
                misTakeCount++;
                cost += -test;
            }
        }

        log.info("ITERATION: {}, MISTAKE_RATE : {}, COST: {}", ITER_COUNT++, misTakeCount / (double) instanceLength, cost);

        return cost;
    }

    @Override
    public <T> void parameterGradient(int start, int end, T theta) {
        gGradient(start, end, theta);
    }

    @Override
    public <T> void gGradient(int start, int end, T theta) {

        for (int i = 0; i < instanceLength; i++) {
            double[] X = data.getInstance(i);
            double y = data.getLabel(i);
            if (hypothesis(X) * y <= 0) {
                m[i] += y;
            }
        }
    }

    private double hypothesis(double[] x) {

        double result = 0;
        for (int i = 0; i < instanceLength; i++)
            if (m[i] != 0) result += m[i] * kernel.similarity(data.getInstance(i), x);

        return result;
    }
}
