package data.builder;

import data.DataSet;
import data.core.Label;
import data.core.SparseMatrix;
import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.sort.SortFloatIntUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by hanxuan on 9/17/15.
 */
public class SparseMatrixDataSetBuilder extends Builder{

    private static Logger log = LogManager.getLogger(SparseMatrixDataSetBuilder.class);

    private HashMap<Object, Integer> classIndexMap;

    public SparseMatrixDataSetBuilder(String path, String sep, boolean hasHeader, boolean needBias, int featureCount,
                                      int instanceCount, int[] categoryIndex, boolean classification) {
        super(path, sep, hasHeader, needBias, featureCount, instanceCount, categoryIndex, classification);
    }

    @Override
    public void build() throws IOException {

        int instanceLength = countLines();
        final int[][] indexes = new int[instanceLength][];
        final float[][] data= new float[instanceLength][];
        final float[] label = new float[instanceLength];
        final boolean[] indicator = new boolean[needBias ? featureCount + 1 : featureCount];
        if (needBias) indicator[0] = true;

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        int pointer = 0;
        while ((line = reader.readLine()) != null) {

            String[] es = line.trim().split(sep);

            if (isClassification) {
                label[pointer] = classIndexMap.get(Integer.parseInt(es[0].trim()));
            }else {
                label[pointer] = Float.parseFloat(es[0].trim());
            }

            int[] featureIndex = new int[es.length - 1];
            float[] dataRow = new float[es.length - 1];
            for (int i = 1; i < es.length; i++) {
                String[] fd = es[i].trim().split(":");
                featureIndex[i - 1] = Integer.parseInt(fd[0]);
                dataRow[i - 1] = Float.parseFloat(fd[1]);
            }

            SortFloatIntUtils.sort(dataRow, featureIndex);

            if (needBias) {
                data[pointer] = new float[es.length];
                indexes[pointer] = new int[es.length];
                data[pointer][0] = 1F;
                System.arraycopy(dataRow, 0, data[pointer], 1, dataRow.length);
                System.arraycopy(featureIndex, 0, indexes[pointer], 1, featureIndex.length);
            }else {
                data[pointer] = dataRow;
                indexes[pointer] = featureIndex;
            }

            pointer++;
        }

        SparseMatrix matrix = new SparseMatrix(indexes, data, indicator.length, indicator);
        Label label1 = new Label(label, classIndexMap);
        dataSet = new DataSet(matrix, label1);
    }

    private int countLines() throws IOException{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        int counter = 0;
        TIntIntHashMap map = new TIntIntHashMap();
        while ((line = reader.readLine()) != null) {
            ++ counter;
            if (isClassification) {
                int label = Integer.parseInt(line.trim().split(sep)[0].trim());
                map.adjustOrPutValue(label, 1, 1);
            }
        }

        int counter2 = 0;
        if (isClassification) {
            classIndexMap = new HashMap<>();
            int[] keys = map.keys();
            Arrays.sort(keys);
            for (int key : keys){
                classIndexMap.put(key, counter2++);
                log.info("class {} has instance {}", key, map.get(key));
            }
        }

        return counter;
    }

    public static void main(String[] args) throws IOException{
        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/train.trec/feature_matrix.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = true;
        int m = 1754;
        int n = 11314;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new SparseMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        log.info(dataset.getInstanceLength());
        log.info(dataset.getFeatureLength());

        log.info("============");

        log.info(dataset.getInstance(0));

        log.info(dataset.getInstance(0)[749]);  // 0.29949814081192017
        log.info(dataset.getInstance(0)[869]);  // 1.3157238960266113
        log.info(dataset.getInstance(1)[1023]); // 0.202371746301651
        log.info(dataset.getInstance(1)[547]);  // 1.187520980834961

        log.info("============");

        log.info(dataset.getEntry(0, 749));
        log.info(dataset.getEntry(0, 869));
        log.info(dataset.getEntry(1, 1023));
        log.info(dataset.getEntry(1, 547));

        log.info("============");

        log.info(dataset.getInstance(dataset.getInstanceLength() - 1)[436]); //436:0.3452489972114563
        log.info(dataset.getInstance(dataset.getInstanceLength() - 1)[565]); //565:0.24571247398853302

        log.info("============");

        log.info(dataset.getEntry(dataset.getInstanceLength() - 1, 436));
        log.info(dataset.getEntry(dataset.getInstanceLength() - 1, 565));

        log.info("============");

        log.info(dataset.getFeatureCol(0)[1]);
        log.info(dataset.getFeatureCol(1)[0]);
        log.info(dataset.getFeatureCol(1)[1]);
        log.info(dataset.getFeatureCol(1)[2]);

        log.info(dataset.getFeatureCol(749)[0]);
        log.info(dataset.getFeatureCol(749)[1]);
        log.info(dataset.getFeatureCol(749)[2]);
        log.info(dataset.getFeatureCol(749)[3]);
        log.info(dataset.getFeatureCol(749)[4]);

    }
}
