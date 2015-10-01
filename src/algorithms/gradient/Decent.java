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

    double COST_DECENT_THRESHOLD = 0.0001;

    int MAX_ROUND = 50000;

    double cost(DataSet data, double[] theta);

    void parameterGradient(DataSet data, int start, int end, double[] theta);

    default double loop(DataSet data, int bucketCount, double[] theta) {

        bucketCount = Math.min(bucketCount, data.getInstanceLength());
        int BUCKET_LENGTH = data.getInstanceLength() / bucketCount;

        double miniCost = Integer.MAX_VALUE;

        for (int ii = 0; ; ++ ii) {

            try{

                int i = ii % bucketCount;

                int start = i * BUCKET_LENGTH;
                int end = Math.min((1 + i) * BUCKET_LENGTH, data.getInstanceLength());

                parameterGradient(data, start, end, theta);
                double cost = cost(data, theta);

                double deltaCost = Math.abs(cost - miniCost);

                if (cost < miniCost){
                    miniCost = cost;
                }

                if (ii % 100 == 0) {

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
