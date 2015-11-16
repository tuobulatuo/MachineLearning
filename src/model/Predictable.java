package model;

/**
 * Created by hanxuan on 9/18/15 for machine_learning.
 */
public interface Predictable {

    double predict(double[] feature);

    default double score(double[] feature){
        return predict(feature);
    }

    default double[] probs(double[] feature){
        System.out.println("default probs method not allowed, exit");
        System.exit(-1);
        return new double[0];
    }

}
