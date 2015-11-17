package feature.pca;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.random.RandomUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/10/15 for machine_learning.
 */
public class PCAMain {

    private static Logger log = LogManager.getLogger(PCA.class);

    public static void main(String[] args) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spam_polluted/allSet";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 1057;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        PCA pca = new PCA(dataset);
        pca.rotate();

        double totalVar = pca.totalVar(dataset);
        double[] pve = new double[dataset.getFeatureLength()];
        IntStream.range(0, pve.length).forEach(j -> pve[j] = pca.varianceExplainedM(j, dataset) / totalVar);
        double[] cpve = new double[pve.length];
        cpve[0] = pve[0];
        for (int j = 1; j < cpve.length; j++) {
            cpve[j] = cpve[j - 1] + pve[j];
        }

        int[] index = RandomUtils.getIndexes(pve.length);

        log.info("{}", index);
        log.info("{}", pve);
        log.info("{}", cpve);
        log.info("==============");

        DataSet principleTopNSet = pca.project(dataset, 200);
        String pcaOut = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spam_polluted/allSet.pca.200";
        BufferedWriter writer = new BufferedWriter(new FileWriter(pcaOut), 1024 * 1024 * 32);
        Map indexClassMap = dataset.getLabels().getIndexClassMap();
        for (int i = 0; i < principleTopNSet.getInstanceLength(); i++) {
            double[] x = principleTopNSet.getInstance(i);
            int label = (int) principleTopNSet.getLabel(i);
            StringBuilder sb = new StringBuilder();
            Arrays.stream(x).forEach(e -> sb.append(e + "\t"));
            sb.append(indexClassMap.get(label));
            writer.write(sb.toString() + "\n");
        }
        writer.close();
    }
}
