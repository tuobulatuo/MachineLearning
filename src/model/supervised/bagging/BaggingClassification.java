package model.supervised.bagging;

import data.DataSet;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class BaggingClassification extends BootstrapAveraging{

    private int classCount = 0;

    @Override
    public double predict(double[] feature) {

        double[] votes = new double[classCount];
        for (int i = 0; i < predictables.length; i++) {
            votes[(int) predictables[i].predict(feature)] += 1;
        }
        int[] indexes = RandomUtils.getIndexes(classCount);
        SortIntDoubleUtils.sort(indexes, votes);
        return indexes[classCount - 1];
    }

    @Override

    public void initialize(DataSet d) {
        super.initialize(d);
        classCount = d.getLabels().getIndexClassMap().size();
    }
}
