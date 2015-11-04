package model.supervised.boosting.gradiantboost.gradientboostor;

import data.DataSet;
import model.supervised.cart.RegressionTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.random.RandomUtils;

import java.util.Arrays;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class GradientRegressionTree extends RegressionTree implements GradientBooster {

    private static Logger log = LogManager.getLogger(GradientRegressionTree.class);

    public GradientRegressionTree() {
        COST_DROP_THRESHOLD = Integer.MIN_VALUE;
    }

    public GradientRegressionTree(int depth, DataSet dataSet, int[] existIds) {

        super(depth, dataSet, existIds);

        double[] labels = Arrays.stream(existIds).mapToDouble(id -> dataSet.getLabel(id)).toArray();
        squareError = cost(labels);

        log.debug("Tree {} @depth {} constructed, squareError: {}, has {} points ...", td, depth, squareError, existIds.length);
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {
        left = new GradientRegressionTree(this.depth + 1, this.dataSet, leftGroup);
        right = new GradientRegressionTree(this.depth + 1, this.dataSet, rightGroup);
    }

    @Override
    public <T> void boostInitialize(DataSet data, T info) {
        this.dataSet = data;
        this.existIds = RandomUtils.getIndexes(dataSet.getInstanceLength());
        log.debug("boostInitialize finished ...");
    }

    @Override
    public void boost() {
        train();
    }

    @Override
    public double boostPredict(double[] feature) {
        return predict(feature);
    }
}
