package model.supervised.cart;

/**
 * Created by hanxuan on 9/18/15 for machine_learning.
 */
public class FeatureThresholdPair {

    private int featureId = 0;

    private double threshold = 0;

    public static final FeatureThresholdPair IN_VALID_PAIR = new FeatureThresholdPair(-1, -1);

    public FeatureThresholdPair(int featureId, double threshold) {
        this.featureId = featureId;
        this.threshold = threshold;
    }

    public boolean isValid() {
        return !this.equals(IN_VALID_PAIR);
    }

    public void setFeatureId(int featureId) {
        this.featureId = featureId;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public int getFeatureId() {
        return featureId;
    }

    public double getThreshold() {
        return threshold;
    }

    public static void main(String[] args) {
        FeatureThresholdPair pair1 = FeatureThresholdPair.IN_VALID_PAIR;
        FeatureThresholdPair pair2 = FeatureThresholdPair.IN_VALID_PAIR;
        FeatureThresholdPair pair3 = FeatureThresholdPair.IN_VALID_PAIR;
        pair3.setThreshold(-1);

        System.out.println(pair1.equals(FeatureThresholdPair.IN_VALID_PAIR));
        System.out.println(pair2.equals(FeatureThresholdPair.IN_VALID_PAIR));
        System.out.println(pair3.equals(FeatureThresholdPair.IN_VALID_PAIR));
        System.out.println(pair3.equals(FeatureThresholdPair.IN_VALID_PAIR));
        System.out.println(pair3 == FeatureThresholdPair.IN_VALID_PAIR);
    }

}
