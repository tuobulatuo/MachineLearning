package model.supervised.cart;

import data.DataSet;
import gnu.trove.map.hash.TDoubleIntHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.sort.SortDoubleDoubleUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public class ClassificationTree extends Tree{

    private static final Logger log = LogManager.getLogger(ClassificationTree.class);

    public static final double INFORMATION_GAIN_THRESHOLD = 0.1;

    private double randomness = Integer.MAX_VALUE;

    public ClassificationTree() {
        super();
    }

    public ClassificationTree(DataSet dataSet) {

        this(0, dataSet, IntStream.range(0, dataSet.getInstanceLength()).toArray());
    }

    public ClassificationTree(int depth, DataSet dataSet, int[] existIds) {

        super(depth, dataSet, existIds);

        TIntIntHashMap counter = new TIntIntHashMap();
        Arrays.stream(existIds).forEach(id -> counter.adjustOrPutValue((int) dataSet.getLabel(id), 1, 1));
        double[] pa = Arrays.stream(counter.values()).mapToDouble(i -> i * 1.0 / existIds.length).toArray();
        randomness = h(pa);

        log.debug("Distribution: {}", pa);
        log.info("Tree {} @depth {} constructed, randomness: {}, has {} points ...", td, depth, randomness, existIds.length);
    }

    @Override
    public double gainByCriteria(double[] labels, int position) {
        return growByInformationGain(labels, position);
    }

    @Override
    protected boolean lessThanImpurityGainThreshold(double gain) {
        return gain <= INFORMATION_GAIN_THRESHOLD;
    }

    @Override
    protected void setTreeLabel() {

        TDoubleIntHashMap counter = new TDoubleIntHashMap();
        Arrays.stream(existIds).forEach(i -> counter.adjustOrPutValue(dataSet.getLabel(i), 1, 1));

        double[] keys = counter.keys();
        double[] values = Arrays.stream(keys).map(k -> counter.get(k)).toArray();

        SortDoubleDoubleUtils.sort(keys, values);
        treeLabel = keys[keys.length - 1];
        log.info("[LEAF NODE] id: {}, label: {}", td, treeLabel);
        log.info("[LEAF NODE] categories: {}", Arrays.toString(keys));
        log.info("[LEAF NODE] counts: {}", Arrays.toString(values));
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

        left = new ClassificationTree(this.depth + 1, this.dataSet, leftGroup);
        right = new ClassificationTree(this.depth + 1, this.dataSet, rightGroup);
    }


    private double growByInformationGain(double[] labels, int position) {

        TIntIntHashMap counterB = new TIntIntHashMap();
        TIntIntHashMap counterC = new TIntIntHashMap();
        Arrays.stream(labels, 0, position).forEach(i -> counterB.adjustOrPutValue((int) i, 1, 1));
        Arrays.stream(labels, position, existIds.length).forEach(i -> counterC.adjustOrPutValue((int) i, 1, 1));

        double[] pb = Arrays.stream(counterB.values()).mapToDouble(i -> i * 1.0 / position).toArray();
        double[] pc = Arrays.stream(counterC.values()).mapToDouble(i -> i * 1.0 / (existIds.length - position)).toArray();

        return randomness * existIds.length - h(pb) * position - h(pc) * (existIds.length - position);
    }

    private double h(double[] p) {
        double loge2 = Math.log(2);
        return Arrays.stream(p).map(i -> -i * Math.log(i) / loge2).sum();
    }

    public static void main(String[] args) {

        ClassificationTree ct = new ClassificationTree(0, null, new int[0]);

        double[] p0 = {0.5, 0.5};
        log.info(ct.h(p0));
    }
}
