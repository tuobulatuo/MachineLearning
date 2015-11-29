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
public class ProcessLaplace extends Process{

    private static Logger log = LogManager.getLogger(ProcessLaplace.class);


    @Override
    public void tableProcess(float[][] table) {

        AtomicInteger rareEvent = new AtomicInteger(0);
        IntStream.range(0, table.length).forEach(i -> {
            if (ArraySumUtil.sum(table[i]) <= table[i].length + rareThreshold) rareEvent.getAndIncrement();
            float[] probs = ArraySumUtil.normalize(table[i]);
            IntStream.range(0, probs.length).forEach(j -> probs[j] = (float) NumericalComputation.logOdds(probs[j]));
        });

        log.info("ProcessLaplace");
        log.info("rare {}", rareEvent.get());
    }
}
