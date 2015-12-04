package project.datapreprocess;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;

import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class PriorNoBuzz extends Process{

    private static Logger log = LogManager.getLogger(PriorNoBuzz.class);

    @Override
    public void tableProcess(float[][] table) {

        float[] backgroundProbs = new float[table[0].length];
        for (int i = 0; i < table.length; i++)
            for (int j = 0; j < backgroundProbs.length; j++)
                backgroundProbs[j] += table[i][j];

        ArraySumUtil.normalize(backgroundProbs);

        for (int i = 0; i < table.length; i++) {
            float[] probs = ArraySumUtil.normalize(table[i]);
            IntStream.range(0, probs.length).forEach(j -> probs[j] = (float) Math.log(probs[j] / backgroundProbs[j]));
        }

        log.info("tableProcess PriorNoBuzz");
        log.debug("backgroundProbs {}", backgroundProbs);
    }
}
