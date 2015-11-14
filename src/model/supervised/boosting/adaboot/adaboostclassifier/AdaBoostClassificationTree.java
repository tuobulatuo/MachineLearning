package model.supervised.boosting.adaboot.adaboostclassifier;

import data.DataSet;
import model.supervised.cart.ClassificationTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public class AdaBoostClassificationTree extends ClassificationTree implements AdaBoostClassifier {

    private static Logger log = LogManager.getLogger(AdaBoostClassificationTree.class);

    public AdaBoostClassificationTree(){}

    public AdaBoostClassificationTree(int depth, DataSet dataSet, int[] existIds){
        super(depth, dataSet, existIds);
    }

    @Override
    protected void split(int[] leftGroup, int[] rightGroup) {

        left = new AdaBoostClassificationTree(this.depth + 1, this.dataSet, leftGroup);
        right = new AdaBoostClassificationTree(this.depth + 1, this.dataSet, rightGroup);
    }

    @Override
    public <T> void boostInitialize(DataSet data, T info) {

        this.dataSet = data;
        this.existIds = (int[]) info;

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
