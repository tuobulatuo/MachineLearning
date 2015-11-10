package data.builder;

import data.DataSet;
import data.core.FullMatrix;
import data.core.Label;
import gnu.trove.impl.sync.TSynchronizedIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 * @author hanxuan
 */
public class FullMatrixDataSetBuilder extends Builder{

    private static final Logger log = LogManager.getLogger(FullMatrixDataSetBuilder.class);

    public FullMatrixDataSetBuilder(String path, String sep, boolean hasHeader, boolean needBias,
                                    int m, int n, int[] categoryIndex, boolean classification){
        super(path, sep, hasHeader, needBias, m, n, categoryIndex, classification);
    }

    @Override
    public void build() throws IOException {

        log.info("start building FullMatrixDataSet ...");

        final AtomicInteger n = new AtomicInteger(0);
        final TSynchronizedIntObjectMap categoryCounter
                = new TSynchronizedIntObjectMap(new TIntObjectHashMap<HashSet<String>>());
        final HashSet<String> classCounter = new HashSet<>();

        for (int i : categoryIndex.toArray()) {
            categoryCounter.put(i, new HashSet<String>());
        }

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 64);

        String[] featureNames = null;
        if (hasHeader) {
            featureNames = reader.readLine().split(sep);
        }

        reader.lines().forEach(
                line -> {
                    line = line.toUpperCase().trim();
                    n.getAndIncrement();
                    if (categoryIndex.size() > 0) {
                        String[] es = line.split(sep);
                        for (int i : categoryIndex.toArray()) {
                            ((Set) categoryCounter.get(i)).add(es[i].trim());
                        }
                    }
                    if (isClassification) {
                        String[] es = line.split(sep);
                        classCounter.add(es[es.length - 1].trim());
                    }
                }
        );
        reader.close();

        int expand = categoryIndex.size();
        for (int key : categoryCounter.keys()) {
            expand += ((Set)categoryCounter.get(key)).size();
        }

        featureCount = featureCount - categoryIndex.size() * 2 + expand + (needBias ? 1 : 0);

        int categoryFeatureStart = featureCount - (expand - categoryIndex.size());
        boolean[] categoryIndicator = new boolean[featureCount];
        if (needBias) categoryIndicator[0] = true;
        Arrays.fill(categoryIndicator, categoryFeatureStart, categoryIndicator.length, true);

        int pos = categoryFeatureStart;
        HashMap<Object, Integer> categoryIndexMap = new HashMap<>();
        for (int key : categoryCounter.keys()) {
            Set<String> set = (Set<String>) categoryCounter.get(key);
            for (String s : set){
                categoryIndexMap.put(""+ key + s, pos++);
            }
        }

        HashMap<Object, Integer> classIndexMap = new HashMap<>();
        int counter = 0;
        for (Object c: classCounter) {
            classIndexMap.put(c, counter++);
        }

        if (instanceCount != n.get()) {
            log.warn("INSTANCE COUNT: given({}), real({})", instanceCount, n.get());
        }
        instanceCount = n.get();

        log.info("Expanded featureCount: {}, instanceCount: {}", featureCount, instanceCount);

        final float[][] data = new float[instanceCount][featureCount];
        IntStream.range(0, instanceCount).forEach(i -> data[i][0] = 1);
        final float[] labelVec = new float[instanceCount];

        reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 64);

        if (hasHeader) reader.readLine();

        n.set(0);
        reader.lines().forEach(
                line -> {
                    line = line.toUpperCase().trim();
                    int currentFeaturePointer = needBias ? 1 : 0;
                    String[] es = line.split(sep);
                    for (int i = 0; i < es.length - 1; i++) {
                        if (categoryIndex.contains(i)) {
                            data[n.get()][categoryIndexMap.get("" + i + es[i].trim())] = 1;
                        } else {
                            data[n.get()][currentFeaturePointer++] = Float.parseFloat(es[i].trim());
                        }
                    }

                    labelVec[n.get()] = isClassification ? classIndexMap.get(es[es.length - 1].trim()) :
                            Float.parseFloat(es[es.length - 1].trim());

                    n.getAndIncrement();

                    if (n.get() % 200000 == 0) {
                        log.info("{} lines processed ... ", n.get());
                    }
                }
        );
        reader.close();

        FullMatrix matrix = new FullMatrix(data, categoryIndicator, featureNames);
        Label label;
        if (isClassification) {
            label = new Label(labelVec, classIndexMap, categoryIndexMap);
        }else {
            label = new Label(labelVec, classIndexMap);
        }

        dataSet = new DataSet(matrix, label);

        log.info("FullMatrixDataSetBuilder work finished ... ");

    }
}
