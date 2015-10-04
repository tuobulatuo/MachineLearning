package model.supervised.neuralnetwork;

import algorithms.gradient.Decent;
import algorithms.gradient.GradientDecent;
import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;
import utils.NumericalComputation;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/1/15 for machine_learning.
 */
public class NeuralNetwork implements Trainable, Predictable, GradientDecent, Decent {

    private static final Logger log = LogManager.getLogger(NeuralNetwork.class);

    public static double COST_DECENT_THRESHOLD = 0.00000001;

    public static int MAX_ROUND = 5000;

    public static int PRINT_GAP = 100;

    public static boolean PRINT_HIDDEN = false;

    public static double EPSILON = 1;

    public static double ALPHA = 0.01;   // learning rate

    public static double LAMBDA = 0.0;  // punish rate

    public static int BUCKET_COUNT = 1;   // mini batch

    private double[][][] theta = null;

    private int[] structure = null;

    private int layerCount = 0;

    private boolean biased = true;

    private DataSet data = null;

    public NeuralNetwork(int[] structure, boolean bias) {
        this.structure = structure;
        this.biased = bias;
    }

    @Override
    public void initialize(DataSet d) {

        this.data = d;

        layerCount = structure.length;
        theta = new double[layerCount - 1][][];
        for (int i = 1; i < layerCount; i++) {

            int layerIn = structure[i - 1];
            int layerOut = structure[i];

            if (i < layerCount - 1) {
                layerOut -= (biased ? 1 : 0);    //  output layer no bias other layer consider bias.
            }

            double[][] w = new double[layerOut][];
            double epsilonInit = EPSILON * Math.sqrt(6 / (double)(layerIn + layerOut));
            int randMin = Math.min(layerIn, layerOut);
            int randMax = Math.max(layerIn, layerOut) + 1;
            for (int j = 0; j < layerOut; j++) {
                w[j] = Arrays.stream(RandomUtils.randomIntRangeArray(randMin, randMax, layerIn)).
                        mapToDouble(x ->  epsilonInit * (x * 2.0  - 1)).toArray();
            }
            theta[i - 1] = w;
        }

        log.info("Initial theta: {}", Arrays.deepToString(theta));

        log.info("Neural Network initialized, with {} layers, theta dimension: {}, bias = {}",
                layerCount, structure, biased);
    }

    @Override
    public double predict(double[] feature) {
        double[] labels = Arrays.stream(feedForward(feature, theta)).map(x -> x * 1000000).toArray();
        int[] index = RandomUtils.getIndexes(labels.length);
        SortIntDoubleUtils.sort(index, labels);
        return index[index.length - 1];
    }

    @Override
    public double score(double[] feature) {
        double[] labels = Arrays.stream(feedForward(feature, theta)).map(x -> x * 1000000).toArray();
        int[] index = RandomUtils.getIndexes(labels.length);
        SortIntDoubleUtils.sort(index, labels);
        return index[index.length - 1] == 1 ? labels[index.length - 1] / (double) 1000000 : 1 - labels[index.length - 1] / (double) 1000000;
    }

    @Override
    public void train() {

        double initialCost = cost(theta);
        log.info("Training started, initialCost: {}", initialCost);
        loop(data.getInstanceLength(), BUCKET_COUNT, theta, COST_DECENT_THRESHOLD, MAX_ROUND, PRINT_GAP);
        log.info("Training finished ...");
    }

    @Override
    public <T> double cost(T params) {

        double[][][] theta = (double[][][]) params;

        AtomicDouble cost = new AtomicDouble(0);

        IntStream.range(0, data.getInstanceLength()).parallel().forEach(
                i -> {
                    double[] X = data.getInstance(i);
                    double[] labels = feedForward(X, theta);
                    double[] ys = yVector(i);
                    double accu = 0;
                    for (int j = 0; j < ys.length; j++) {
                        accu += -(ys[j] * Math.log(labels[j]) + (1 - ys[j]) * Math.log(1 - labels[j]));
                    }
                    cost.getAndAdd(accu);
                }
        );

        AtomicDouble punish = new AtomicDouble(0);
        IntStream.range(0, theta.length).forEach(i -> {
            double[][] currentTheta = theta[i];
            for (int j = 0; j < currentTheta.length; j++) {
                for (int k = 1; k < currentTheta[0].length; k++) {
                    punish.getAndAdd(Math.pow(currentTheta[j][k], 2));
                }
            }
        });

        return cost.get() + punish.get() * LAMBDA;
    }

    @Override
    public <T> void gGradient(int start, int end, T params) {

        double[][][] theta = (double[][][]) params;
        double[][][] gradient = new double[theta.length][][];
        for (int i = 0; i < theta.length; i++) {
            gradient[i] = new double[theta[i].length][theta[i][0].length];
        }

        IntStream.range(start, end).parallel().forEach(
                i -> backPropagation(i, gradient)
        );

        IntStream.range(0, theta.length).forEach(
                i -> {
                    double[][] currentLayerTheta = theta[i];
                    IntStream.range(0, currentLayerTheta.length).parallel().forEach(
                            j -> {
                                double[] w = currentLayerTheta[j];
                                for (int k = 0; k < w.length; k++) {
                                    w[k] -= ALPHA * gradient[i][j][k];
                                }
                            }
                    );
                }
        );

    }

    private void backPropagation(int i, double[][][] gradient) {

        double[] X = data.getInstance(i);
        double[] yVector = yVector(i);

        double[][] Z = new double[layerCount][];
        double[][] A = new double[layerCount][];
        A[0] = X;

        for (int j = 1; j < layerCount; j++) {
            double[][] currentLayerTheta = theta[j - 1];

            Z[j] = new double[currentLayerTheta.length];
            for (int k = 0; k < currentLayerTheta.length; k++) {
                double[] w = currentLayerTheta[k];
                Z[j][k] = z(A[j - 1], w);
            }

            double[] AZ = a(Z[j]);
            if (j < layerCount - 1 && biased) {
                int activeNodeLength = Z[j].length + 1;
                A[j] = new double[activeNodeLength];
                A[j][0] = 1;
                System.arraycopy(AZ, 0, A[j], 1, AZ.length);
            } else {
                A[j] = AZ;
            }
        }

        double[][] DELTA = new double[layerCount][];
        DELTA[layerCount - 1] = IntStream.range(0, yVector.length).mapToDouble(
                idx -> A[layerCount - 1][idx] - yVector[idx])
                .toArray();
        for (int j = layerCount - 2; j >= 1; --j) {

            double[][] currentLayerTheta = theta[j];
            double[] currentZ = Z[j];
            double[] sigmG;
            if (biased) {
                double[] ZBias = new double[currentZ.length + 1];
                ZBias[0] = 1;
                System.arraycopy(currentZ, 0, ZBias, 1, currentZ.length);
                sigmG = Arrays.stream(ZBias).map(x -> NumericalComputation.sigmoidGradient(x)).toArray();
            }else {
                sigmG = Arrays.stream(currentZ).map(x -> NumericalComputation.sigmoidGradient(x)).toArray();
            }

            DELTA[j] = new double[currentLayerTheta[0].length];
            for (int k = 0; k < DELTA[j].length; k++) {
                int col = k;
                double[] w = IntStream.range(0, currentLayerTheta.length).mapToDouble(x -> currentLayerTheta[x][col]).toArray();
                DELTA[j][k] = z(DELTA[j + 1], w) * sigmG[k];
            }

            if (biased) {
                double[] DELTAremove0 = new double[DELTA[j].length - 1];
                System.arraycopy(DELTA[j], 1, DELTAremove0, 0, DELTAremove0.length);
                DELTA[j] = DELTAremove0;
            }
        }

        for (int j = 0; j < gradient.length; j++) {
            double[][] currentG = gradient[j];
            for (int k = 0; k < currentG.length; k++) {
                double[] g = currentG[k];
                synchronized (g) {
                    for (int l = 0; l < g.length; l++) {
                        g[l] += DELTA[j + 1][k] * A[j][l];
                        g[l] += l > 0 ? LAMBDA * theta[j][k][l] : 0;
                    }
                }
            }

        }
    }

    public double[] feedForward(double[] feature, double[][][] theta) {

        double[] X = feature;

        double[][] Z = new double[layerCount][];
        double[][] A = new double[layerCount][];
        A[0] = X;

        for (int j = 1; j < layerCount; j++) {
            double[][] currentLayerTheta = theta[j - 1];

            Z[j] = new double[currentLayerTheta.length];
            for (int k = 0; k < currentLayerTheta.length; k++) {
                double[] w = currentLayerTheta[k];
                Z[j][k] = z(A[j - 1], w);
            }

            double[] AZ = a(Z[j]);

            log.debug("HIDDEN Z: {}-{}", j, Z[j]);
            log.debug("HIDDEN A: {}-{}", j, AZ);

            if (j < layerCount - 1 && biased) {
                int activeNodeLength = Z[j].length + 1;
                A[j] = new double[activeNodeLength];
                A[j][0] = 1;
                System.arraycopy(AZ, 0, A[j], 1, AZ.length);
            } else {
                A[j] = AZ;
            }
        }

        if (PRINT_HIDDEN) {
            for (int i = 1; i < A.length - 1; i++) {
                System.out.println("HIDDEN " + i + " : " + Arrays.toString(A[i]));
            }
        }

        return A[A.length - 1];
    }

    private double[] a(double[] Z) {
        return Arrays.stream(Z).map(x -> NumericalComputation.sigmoid(x)).toArray();
    }

    private double z(double[] A, double[] theta){
        return IntStream.range(0, A.length).mapToDouble(i -> A[i] * theta[i]).sum();
    }

    @Override
    public <T> void parameterGradient(int start, int end, T params) {
        gGradient(start, end, params);
    }

    public double[] yVector(int idx) {
        double y = data.getLabel(idx);
        double[] yVector = new double[structure[structure.length - 1]];
        yVector[(int) y] = 1;
        return yVector;
    }

    public double[][][] getTheta(){return theta;}

    public static void main(String[] args) {
        int[] struct = new int[]{6, 4, 2, 1};
        NeuralNetwork mp = new NeuralNetwork(struct, false);
        mp.initialize(null);
        log.info(Arrays.deepToString(mp.theta));
    }
}
