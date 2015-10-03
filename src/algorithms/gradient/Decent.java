package algorithms.gradient;

import data.DataSet;
import model.supervised.linearmodel.LinearGradientDecent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


/**
 * Created by hanxuan on 9/30/15 for machine_learning.
 */
public interface Decent {

    Logger log = LogManager.getLogger(Decent.class);

    double COST_DECENT_THRESHOLD = 0.0000000;

    int MAX_ROUND = 1000000;

    <T> double cost(T theta);

    <T> void parameterGradient(int start, int end, T theta);

    default <T> double loop(int instanceLength, int bucketCount, T theta) {

        bucketCount = Math.min(bucketCount, instanceLength);
        int BUCKET_LENGTH = instanceLength / bucketCount;

        double miniCost = Integer.MAX_VALUE;

        for (int ii = 0; ; ++ ii) {

            try{

                int i = ii % bucketCount;

                int start = i * BUCKET_LENGTH;
                int end = Math.min((1 + i) * BUCKET_LENGTH, instanceLength);

                parameterGradient(start, end, theta);
                double cost = cost(theta);

                double deltaCost = Math.abs(cost - miniCost);

                if (cost < miniCost){
                    miniCost = cost;
                }

                if (ii % 10000 == 0) {

                    log.info("round {}, cost {}, theta {}", ii, cost, theta);
                    if (deltaCost < COST_DECENT_THRESHOLD || ii > MAX_ROUND) {
                        return miniCost;
                    }
                }

            }catch (Throwable t) {
                log.error(t.getMessage(), t);
            }
        }
    }
}
