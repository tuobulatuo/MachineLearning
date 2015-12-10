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

    public SelectNeighborBy selectNeighborBy = SelectNeighborBy.RANK;

    public EstimateBy estimateBy = EstimateBy.NEIGHBOR;

    private static Logger log = LogManager.getLogger(KNN.class);

    private DataSet data = null;

    private int classCount = 0;

    private int instanceLength = 0;

    private Kernel kernel = null;

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

        if(estimateBy == EstimateBy.NEIGHBOR) {
            return probsByNeighbor(feature);
        }else if (estimateBy == EstimateBy.DENSITY) {
            return probsByDensity(feature);
        }else {
            log.error("unknown estimate {}", estimateBy);
            return new double[0];
        }
    }

    private double[] probsByNeighbor(double[] feature) {

        double[] distance = neighborDistance(feature);
        int[] indices = RandomUtils.getIndexes(instanceLength);

        SortIntDoubleUtils.sort(indices, distance); // small -> big, ascending

        int[] topNeighbor = null;
        if (selectNeighborBy == SelectNeighborBy.RADIUS) {
            topNeighbor = getTopNeighbourByRadius(indices, distance);
        }else if (selectNeighborBy == selectNeighborBy.RANK) {
            topNeighbor = getTopNeighbourByRank(indices);
        }

        if (topNeighbor.length == 0) {
            log.warn("topNeighbor.length == 0");
        }

        double[] probs = new double[classCount];
        for (int i = 0; i < topNeighbor.length; i++) {
            probs[(int) data.getLabel(i)] ++;
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
    // so we shift all the numbers to 0.

        double min = Arrays.stream(probs).min().getAsDouble();
        for (int i = 0; i < classCount; i++) probs[i] = probs[i] - min;
        ArraySumUtil.normalize(probs);
        return probs;
    }

    @Override
    public void train() {
        log.info("~ KNN is ready ~");
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
        while (pointer < R && pointer < indices.length) topR.add(indices[pointer++]);
        return topR.toArray();
    }

    private int[] getTopNeighbourByRadius(int[] indices, double[] distances) {

        TIntArrayList withinR = new TIntArrayList(instanceLength / 100);
        int pointer = 0;
        while (distances[pointer] <= R && pointer < indices.length) withinR.add(indices[pointer++]);
        return withinR.toArray();
    }
}
