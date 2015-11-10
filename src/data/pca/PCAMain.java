package data.pca;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.CrossValidationEvaluator;
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

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.all.txt";
        String sep = "\t";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 51;
        int n = 870000;
        int[] featureCategoryIndex = {0, 1, 2, 3, 4, 5, 6, 7};
        boolean isClassification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet dataset = builder.getDataSet();
        dataset.meanVarianceNorm();

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 100);
        TIntHashSet trainIndexes = new TIntHashSet(kFoldIndex[0]);
        DataSet trainSet = dataset.subDataSetByRow(trainIndexes.toArray());
        trainSet.meanVarianceNorm();

        PCA pca = new PCA(trainSet);
        pca.rotate();

        double totalVar = pca.totalVar(trainSet);
        double[] pve = new double[trainSet.getFeatureLength()];
        IntStream.range(0, pve.length).forEach(j -> pve[j] = pca.varianceExplainedM(j, trainSet) / totalVar);
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


        double totalVar2 = pca.totalVar(dataset);
        IntStream.range(0, pve.length).parallel().forEach(j -> pve[j] = pca.varianceExplainedM(j, dataset) / totalVar2);
        cpve[0] = pve[0];
        for (int j = 1; j < cpve.length; j++) {
            cpve[j] = cpve[j - 1] + pve[j];
        }

        log.info("{}", pve);
        log.info("{}", cpve);
        log.info("\n");

        DataSet principleTopNSet = pca.project(dataset, 27);
        String pcaOut = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.all.pca.txt";
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
