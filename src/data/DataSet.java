package data;

import data.core.Label;
import data.core.AMatrix;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */

public class DataSet {

    private static Logger log = LogManager.getLogger(DataSet.class);

    private AMatrix featureMatrix = null;

    private Label labels = null;

    public DataSet(AMatrix m, Label l) {

        this.featureMatrix = m;
        this.labels = l;
    }

    public DataSet subDataSetByRow(int[] rowIndexes) {
        AMatrix subMatrix = featureMatrix.subMatrixByRow(rowIndexes);
        Label subLabels = labels.subLabelByRow(rowIndexes);
        return new DataSet(subMatrix, subLabels);
    }

    public DataSet balancedDataSet(int sampleSizeMax, int sampleSizeMin) {

        TIntArrayList ids = new TIntArrayList(getInstanceLength());
        for (int i = 0; i < labels.getClassIndexMap().size(); i++) {
            int classIndex = i;
            int[] indexes = IntStream.range(0, getInstanceLength()).filter(j -> (int) getLabel(j) == classIndex).toArray();
            if (indexes.length >= sampleSizeMin){
                TIntArrayList list = new TIntArrayList(indexes);
                list.shuffle(new Random());
                int j = 0;
                for (; j < indexes.length && j < sampleSizeMax; j++) {
                    ids.add(list.get(j));
                }
                log.info("balance sampling class {} | instances {} | sampling {}", i, indexes.length, j);
            }else {
                double[] probs = new double[indexes.length];
                Arrays.fill(probs, 1 / (double) indexes.length);
                EnumeratedIntegerDistribution integerDistribution = new EnumeratedIntegerDistribution(indexes, probs);
                ids.addAll(integerDistribution.sample(sampleSizeMin));
                log.info("balance sampling class {} | instances {} | sampling {}", i, indexes.length, sampleSizeMin);
            }
        }
        log.info("balance sampling total {}", ids.size());

        return subDataSetByRow(ids.toArray());
    }

    public void shiftCompressNorm() {
        featureMatrix.shiftCompressNormalize();
    }

    public void meanVarianceNorm(float[] mean, float[] sd) {
        featureMatrix.meanVarianceNormalize(mean, sd);
    }

    public void meanVarianceNorm() {
        featureMatrix.meanVarianceNormalize();
    }

    public double getEntry(int i, int j){

        return featureMatrix.getEntry(i, j);
    }

    public double[] getInstance(int i) {

        return featureMatrix.getRow(i);
    }

    public double[] getFeatureCol(int featureIndex) {
        return featureMatrix.getCol(featureIndex);
    }

    public double[] getFeatureColFilteredByLabel(int col, Set<Integer> ls) {

        TDoubleArrayList filterFeature = new TDoubleArrayList(getInstanceLength());
        double[] feature = featureMatrix.getCol(col);
        for (int i = 0; i < feature.length; i++) {
            if (ls.contains((int) getLabel(i))) filterFeature.add(feature[i]);
        }
        return filterFeature.toArray();
    }

    public double getLabel(int i) {

        return labels.getRow(i);
    }

    public double getCategoryProportion(int category) {
        return labels.getLabelQuotient(category);
    }

    public AMatrix getFeatureMatrix() {

        return featureMatrix;
    }

    public int getInstanceLength() {

        return featureMatrix.getInstanceLength();
    }

    public int getFeatureLength() {

        return featureMatrix.getFeatureLength();
    }

    public Label getLabels() {

        return labels;
    }

    public int getClassCount() {
        return labels.getClassIndexMap().size();
    }
}
