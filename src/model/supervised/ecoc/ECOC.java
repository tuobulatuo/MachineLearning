package model.supervised.ecoc;

import data.DataSet;
import data.core.Label;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.random.RandomUtils;
import utils.sort.SortIntIntUtils;

import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public abstract class ECOC implements Predictable, Trainable{

    public static int DEFAULT_CODE_WORD_LENGTH = 20;

    public static int USE_EXHAUSTIVE_LIMIT = 7;

    public static int MAX_THREADS = 4;

    private static Logger log = LogManager.getLogger(ECOC.class);

    private Predictable[] predictables = null;

    protected Trainable[] trainables = null;

    private DataSet data = null;

//    private Label[] labels = null;

    private int[][] table = null;

    private int codeWordLength = DEFAULT_CODE_WORD_LENGTH;

    private int classCount = Integer.MIN_VALUE;

    private ExecutorService service = null;

    private CountDownLatch countDownLatch = null;

    public ECOC() {}



    // *****************************

    public abstract void configTrainable();

    // *****************************



    @Override
    public double predict(double[] feature) {

        int[] classError = new int[classCount];
        for (int i = 0; i < classCount; i++) {
            int[] predicts = IntStream.range(0, codeWordLength).map(j -> (int) predictables[j].predict(feature)).toArray();
            classError[i] = codeErrorRate(table[i], predicts);
        }
        int[] indexes = RandomUtils.getIndexes(classCount);
        SortIntIntUtils.sort(indexes, classError);
        return indexes[0];
    }

    @Override
    public void train() {

        service = Executors.newFixedThreadPool(MAX_THREADS);
        countDownLatch = new CountDownLatch(codeWordLength);
        IntStream.range(0, codeWordLength).forEach(i -> {
            service.submit(() -> {
                try{
                    Label label = makeLabels(i);
                    DataSet dataI = new DataSet(data.getFeatureMatrix(), label);
                    Trainable model = trainables[i];
                    model.initialize(dataI);
                    model.train();
                    predictables[i] = model.offer();
                    log.info("ECOC model {} training finished...", i);
                }catch (Throwable t) {
                    log.error(t.getMessage(), t);
                }
                countDownLatch.countDown();
            });
        });

        try {
            TimeUnit.SECONDS.sleep(10);
            countDownLatch.await();
        }catch (Throwable t){
            log.error(t.getMessage(), t);
        }
        service.shutdown();

        log.info("ECOC training finished ... ");
    }

    @Override
    public void initialize(DataSet d) {
        this.data = d;
        classCount = d.getLabels().getIndexClassMap().size();
        if (classCount <= USE_EXHAUSTIVE_LIMIT) {
            codeWordLength = (int) Math.pow(2, classCount - 1) - 1;
        }else {
            codeWordLength = DEFAULT_CODE_WORD_LENGTH;
        }
        table = new int[classCount][codeWordLength];
        predictables = new Predictable[codeWordLength];
        trainables = new Trainable[codeWordLength];

        configTrainable();

        makeTable();

        log.info("ECOC initialized, class count: {}, code word length: {}", classCount, codeWordLength);
    }

    private Label makeLabels(int index) {
        TIntIntHashMap map = new TIntIntHashMap();
        for (int i = 0; i < table.length; i++) {
            map.put(i, table[i][index]);
        }
        float[] vector = new float[data.getInstanceLength()];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = map.get((int)data.getLabel(i));
        }
        HashMap<Object, Integer> classIndexMap = new HashMap<>();
        classIndexMap.put(0, 0);
        classIndexMap.put(1, 1);
        return new Label(vector, classIndexMap);
    }

    private void makeTable() {

        TIntHashSet seen = new TIntHashSet();
        int[] indexes = new int[]{0, 1};
        double[] probs = new double[]{0.5, 0.5};
        EnumeratedIntegerDistribution integerDistribution = new EnumeratedIntegerDistribution(indexes, probs);
        int counter = 0;
        while (counter < classCount) {
            int[] row = integerDistribution.sample(codeWordLength);
            int hashCode = row.hashCode();
            if (!seen.contains(hashCode)) {
                seen.add(hashCode);
                table[counter] = row;
                counter ++;
            }
        }

        log.info("ECOC Table created ...");
        log.info("================== TABLE ==================");
        for (int i = 0; i < table.length; i++) {
            log.info("{}", i, table[i]);
        }
        log.info("================== ===== ==================");
    }

    private int codeErrorRate(int[] result, int[] predicts) {

        int error = 0;
        for (int i = 0; i < result.length; i++) {
            error += result[i] == predicts[i] ? 0 : 1;
        }
        return error;
    }
}
