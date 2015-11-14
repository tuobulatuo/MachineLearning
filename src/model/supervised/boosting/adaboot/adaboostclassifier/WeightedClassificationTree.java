package model.supervised.boosting.adaboot.adaboostclassifier;

import data.DataSet;

import gnu.trove.map.hash.TDoubleDoubleHashMap;

import model.supervised.cart.ClassificationTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;
import utils.random.RandomUtils;
import utils.sort.SortDoubleDoubleUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public class WeightedClassificationTree extends ClassificationTree implements AdaBoostClassifier {

    private static Logger log = LogManager.getLogger(WeightedClassificationTree.class);

    protected double[] weights = null;

    public WeightedClassificationTree(){}

    public WeightedClassificationTree(int depth, DataSet dataSet, int[] existIds, double[] weights){

        super(depth, dataSet, existIds);
        this.weights = weights;
        TDoubleDoubleHashMap counter = new TDoubleDoubleHashMap();
        Arrays.stream(existIds).forEach(id -> counter.adjustOrPutValue(dataSet.getLabel(id), weights[id], weights[id]));
        double[] pa = counter.values();
        randomness = h(pa);

        log.debug("Tree WEIGHTED randomness: {}, tree depth restricted to {}", randomness, MAX_DEPTH);
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

        left = new WeightedClassificationTree(this.depth + 1, this.dataSet, leftGroup, weights);
        right = new WeightedClassificationTree(this.depth + 1, this.dataSet, rightGroup, weights);
    }

    @Override
    public double gainByCriteria(double[] labels, int position, int[] sortedIds) {
        return weightedInformationGain(labels, position, sortedIds);
    }

    private double weightedInformationGain(double[] labels, int position, int[] sortedIds) {

        TDoubleDoubleHashMap counterB = new TDoubleDoubleHashMap();
        TDoubleDoubleHashMap counterC = new TDoubleDoubleHashMap();
        IntStream.range(0, position).forEach(
                i -> counterB.adjustOrPutValue(labels[i], weights[sortedIds[i]], weights[sortedIds[i]]));
        IntStream.range(position, existIds.length).forEach(
                i -> counterC.adjustOrPutValue(labels[i], weights[sortedIds[i]], weights[sortedIds[i]]));

        double[] pb = ArraySumUtil.normalize(counterB.values());
        double[] pc = ArraySumUtil.normalize(counterC.values());

        return randomness * existIds.length - h(pb) * position - h(pc) * (existIds.length - position);
    }

    @Override
    protected void setTreeLabel() {

        TDoubleDoubleHashMap counter = new TDoubleDoubleHashMap();
        Arrays.stream(existIds).forEach(i -> counter.adjustOrPutValue(dataSet.getLabel(i), weights[i], weights[i]));

        double[] keys = counter.keys();
        double[] values = Arrays.stream(keys).map(k -> counter.get(k)).toArray();

        SortDoubleDoubleUtils.sort(keys, values);
        meanResponse = keys[keys.length - 1];
        log.debug("[LEAF NODE] id: {}, label: {}", td, meanResponse);
        log.debug("[LEAF NODE] categories: {}", Arrays.toString(keys));
        log.debug("[LEAF NODE] counts: {}", Arrays.toString(values));
    }

    @Override
    public <T> void boostInitialize(DataSet data, T info) {

        this.dataSet = data;
        this.weights = (double[]) info;
        this.existIds = RandomUtils.getIndexes(data.getInstanceLength());

        TDoubleDoubleHashMap counter = new TDoubleDoubleHashMap();
        Arrays.stream(existIds).forEach(id -> counter.adjustOrPutValue(dataSet.getLabel(id), weights[id], weights[id]));
        double[] pa = counter.values();
        randomness = h(pa);
        log.debug("BoostInitialize finished, Tree WEIGHTED randomness: {}", randomness);
    }

    @Override
    public void boost() {
        train();
    }

    @Override
    public double boostPredict(double[] feature) {
        return predict(feature);
    }

    @Override
    public int bestFeatureId() {
        return featureId;
    }
}
