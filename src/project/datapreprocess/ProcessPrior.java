package project.datapreprocess;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.NumericalComputation;
import utils.array.ArraySumUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class ProcessPrior extends Process{

    private static Logger log = LogManager.getLogger(ProcessPrior.class);

    @Override
    public void tableProcess(float[][] table) {

        double[] defaultProbs = new double[table[0].length];
        IntStream.range(0, table.length).forEach(i ->
                IntStream.range(0, table[i].length).forEach(j -> defaultProbs[j] += table[i][j]));
        ArraySumUtil.normalize(defaultProbs);
        IntStream.range(0, defaultProbs.length).forEach(i -> defaultProbs[i] = NumericalComputation.logOdds(defaultProbs[i]));

        AtomicInteger rareEvent = new AtomicInteger(0);
        IntStream.range(0, table.length).forEach(i -> {
            if (ArraySumUtil.sum(table[i]) <= table[i].length + rareThreshold) {
                System.arraycopy(defaultProbs, 0, table[i], 0, defaultProbs.length);
                rareEvent.getAndIncrement();
            }else {
                float[] probs = ArraySumUtil.normalize(table[i]);
                IntStream.range(0, probs.length).forEach(j -> probs[j] = (float) NumericalComputation.logOdds(probs[j]));
            }
        });

        log.info("ProcessPrior");
        log.info("rare {}", rareEvent.get());
        log.debug("default {}", defaultProbs);
    }
}
