package data.builder;

import data.DataSet;
import data.core.AMatrix;
import data.core.Label;
import gnu.trove.set.hash.TIntHashSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.random.RandomUtils;
//import org.neu.util.rand.RandomUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15 for machine_learning.
 */
public abstract class Builder {

    private static final Logger log = LogManager.getLogger(Builder.class);

    protected String path = null;

    protected String sep = "\\s+";

    protected boolean hasHeader = false;

    protected boolean needBias = true;

    protected TIntHashSet categoryIndex = null;

    protected int featureCount = -1;

    protected int instanceCount = -1;

    protected boolean isClassification = false;

    protected DataSet dataSet = null;

    public Builder(String path, String sep, boolean hasHeader, boolean needBias, int featureCount, int instanceCount, int[] categoryIndex,
                                    boolean classification){

        this.path = path;
        this.sep = sep;
        this.hasHeader = hasHeader;
        this.needBias = needBias;
        this.categoryIndex = new TIntHashSet(categoryIndex);
        this.featureCount = featureCount;
        this.instanceCount = instanceCount;
        this.isClassification = classification;
    }

    public abstract void build() throws IOException;

    public DataSet getDataSet() {
        return dataSet;
    }

    public static void main(String[] args) throws IOException {

        String path0 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/vote/vote.test.data";
        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/homework/hw1/house.test.txt";
        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/homework/hw1/spambase/spambase.data";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 100;
        int[] featureCategoryIndex = RandomUtils.getIndexes(m);

        Builder builder =
                new FullMatrixDataSetBuilder(path0, sep, hasHeader, needBias, m, n, featureCategoryIndex, true);

        builder.build();

        DataSet dataset = builder.getDataSet();
//        dataset.meanVarianceNorm();
        AMatrix ma = dataset.getFeatureMatrix();

        log.info("getFeatureLength: {}", ma.getFeatureLength());
        log.info("getInstanceLength: {}", ma.getInstanceLength());
        log.info("getFeatureNames: {}", Arrays.toString(ma.getFeatureNames()));
        log.info("getBooleanColumnIndicator: {}", Arrays.toString(ma.getBooleanColumnIndicator()));


        IntStream.range(0, ma.getInstanceLength()).forEach(i -> log.info("row{}: {}", i, ma.getRow(i)));
//        IntStream.range(0, ma.getFeatureLength()).forEach(i -> log.info("{} Mean:{}", i, ma.colMean(i)));

        Label l = dataset.getLabels();
        log.info("getClassIndexMap: {}", l.getClassIndexMap());
        log.info("getVector: {}", l.getVector());

    }
}
