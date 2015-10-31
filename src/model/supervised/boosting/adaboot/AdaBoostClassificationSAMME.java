package model.supervised.boosting.adaboot;

import data.DataSet;
import model.Predictable;
import model.Trainable;
//import model.supervised.cart.Tree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public class AdaBoostClassificationSAMME implements Trainable, Predictable{

    private static Logger log = LogManager.getLogger(AdaBoostClassificationSAMME.class);

    private DataSet data = null;

    private double[] alpha = null;

    private double[] weights = null;

    private AdaBoostClassifier[] adaboostClassifier = null;

    public AdaBoostClassificationSAMME(AdaBoostClassifier[] classifiers){
        alpha = new double[classifiers.length];
        adaboostClassifier = classifiers;
    }


    @Override
    public double predict(double[] feature) {
        return 0;
    }

    @Override
    public void train() {

        for (int i = 0; i < alpha.length; i++) {
//            AdaBoostClassifier classifier = new WeightedClassificationTree();
        }

        log.info("AdaBoostClassificationSAMME Training finished ...");
    }

    @Override
    public void initialize(DataSet d) {

        data = d;

        int instanceLength = data.getInstanceLength();
        weights = new double[instanceLength];
        Arrays.fill(weights, 1 / (double) instanceLength);

    }
}
