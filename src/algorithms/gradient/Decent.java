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

    <T> double cost(T theta);

    <T> void parameterGradient(int start, int end, T theta);

    default <T> double loop(int instanceLength, int bucketCount, T theta, double costDecentThreshold, int maxRound,
                            int printGap) {

        bucketCount = Math.min(bucketCount, instanceLength);
        int BUCKET_LENGTH = instanceLength / bucketCount;

        double miniCost = Integer.MAX_VALUE;

        for (int ii = 0; ; ++ ii) {

            try{

                int i = ii % bucketCount;

                int start = i * BUCKET_LENGTH;
                int end = Math.min((1 + i) * BUCKET_LENGTH, instanceLength);

                parameterGradient(start, end, theta);

                if (ii % printGap == 0) {

                    double cost = cost(theta);

                    double deltaCost = Math.abs(cost - miniCost);

                    if (cost < miniCost){
                        miniCost = cost;
                    }

                    log.info("round {}, cost {}", ii, cost);
                    log.debug("theta: {}", theta);
                    if (deltaCost < costDecentThreshold || ii >= maxRound) {
                        return miniCost;
                    }
                }

            }catch (Throwable t) {
                log.error(t.getMessage(), t);
            }
        }
    }
}
