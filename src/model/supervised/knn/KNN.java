package model.supervised.knn;

import data.DataSet;
import gnu.trove.list.array.TIntArrayList;
import model.Predictable;
import model.Trainable;
import model.supervised.kernels.Kernel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;
import utils.random.RandomUtils;
import utils.sort.SortIntDoubleUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 12/10/15 for machine_learning.
 */
public class KNN implements Predictable, Trainable{

    public static double R = 0;

    public static SelectNeighborBy SELECT_NEIGHBOR_BY = SelectNeighborBy.RANK;

    public static EstimateBy ESTIMATE_BY = EstimateBy.NEIGHBOR;

    private static Logger log = LogManager.getLogger(KNN.class);

    private DataSet data = null;

    private int classCount = 0;

    private int instanceLength = 0;

    private Kernel kernel = null;

    private int zeroNeighborCounter = 0;

    public KNN (String kernelClassName) throws Exception{
        kernel = (Kernel) Class.forName(kernelClassName).newInstance();
        log.info("KNN use kernel {}", kernelClassName);
    }

    @Override
    public double predict(double[] feature) {

        double[] probs = probs(feature);
        int[] indices = RandomUtils.getIndexes(probs.length);
        SortIntDoubleUtils.sort(indices, probs);
        return indices[indices.length - 1];
    }

    @Override
    public double[] probs(double[] feature) {

        if(ESTIMATE_BY == EstimateBy.NEIGHBOR) {
            return probsByNeighbor(feature);
        }else if (ESTIMATE_BY == EstimateBy.DENSITY) {
            return probsByDensity(feature);
        }else {
            log.error("unknown estimate {}", ESTIMATE_BY);
            return new double[0];
        }
    }

    private double[] probsByNeighbor(double[] feature) {

        double[] distance = neighborDistance(feature);
        int[] indices = RandomUtils.getIndexes(instanceLength);

        SortIntDoubleUtils.sort(indices, distance); // small -> big, ascending

        log.debug("max dis {}, min dis {}", distance[distance.length -1], distance[0]);

        int[] topNeighbor = null;
        if (SELECT_NEIGHBOR_BY == SelectNeighborBy.RADIUS) {
            topNeighbor = getTopNeighbourByRadius(indices, distance);
        } else if (SELECT_NEIGHBOR_BY == SELECT_NEIGHBOR_BY.RANK) {
            topNeighbor = getTopNeighbourByRank(indices);
        }

        if (topNeighbor.length == 0) {
            log.debug("topNeighbor.length == 0");
            zeroNeighborCounter++;
        }

        double[] probs = new double[classCount];
        for (int i = 0; i < topNeighbor.length; i++) {
            probs[(int) data.getLabel(topNeighbor[i])] ++;
        }
        ArraySumUtil.normalize(probs);
        return probs;
    }

    private double[] probsByDensity(double[] feature) {

        double[] neighborSimilarity = neighborSimilarity(feature);

        double[] probs = new double[classCount];
        for (int i = 0; i < instanceLength; i++) {
            probs[(int) data.getLabel(i)] += neighborSimilarity[i];
        }

    // similarity is not always positive, so this probability is not a probability strictly.
    // probs may appear negative numbers, use with caution.
    // so we shift all the numbers to above 0

        double min = Arrays.stream(probs).min().getAsDouble();
        for (int i = 0; i < classCount; i++) probs[i] = probs[i] - min;
        ArraySumUtil.normalize(probs);
        return probs;
    }

    @Override
    public void train() {
        log.info("~ KNN is too lazy to train ~");
    }

    @Override
    public void initialize(DataSet d) {
        this.data = d;
        classCount = data.getClassCount();
        instanceLength = data.getInstanceLength();
    }

    private double[] neighborDistance(double[] x) {
        double[] distance = new double[instanceLength];
        for (int i = 0; i < instanceLength; i++) {
            distance[i] = - kernel.similarity(data.getInstance(i), x);
        }
        return distance;
    }

    private double[] neighborSimilarity(double[] x) {
        double[] similarity = new double[instanceLength];
        IntStream.range(0, instanceLength).forEach(i -> similarity[i] = kernel.similarity(x, data.getInstance(i)));
        return similarity;
    }

    private int[] getTopNeighbourByRank(int[] indices) {

        TIntArrayList topR = new TIntArrayList((int) R);
        int pointer = 0;
        while (pointer < indices.length && pointer < R) topR.add(indices[pointer++]);
        return topR.toArray();
    }

    private int[] getTopNeighbourByRadius(int[] indices, double[] distances) {

        TIntArrayList withinR = new TIntArrayList(instanceLength / 100);
        int pointer = 0;
        while (pointer < indices.length && distances[pointer] <= R) withinR.add(indices[pointer++]);
        return withinR.toArray();
    }
}
