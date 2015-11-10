package data.pca;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.core.FullMatrix;
import gnu.trove.set.hash.TIntHashSet;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
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
 * Created by hanxuan on 11/9/15 for machine_learning.
 */
public class PCA {

    private static Logger log = LogManager.getLogger(PCA.class);

    private DataSet data = null;

    private EigenDecomposition eigenDecomposition = null;

    private Covariance covariance = null;

    private RealMatrix matrix = null;

    public PCA(DataSet dataSet) {
        data = dataSet;
    }

    public void rotate() {

        long t1 = System.currentTimeMillis();

        int instanceLength = data.getInstanceLength();
        double[][] dataArray = new double[instanceLength][];
        IntStream.range(0, instanceLength).forEach(i -> dataArray[i] = data.getInstance(i));

        matrix = new Array2DRowRealMatrix(dataArray, false);

        log.info("start calculate covariance matrix ...");

        covariance = new Covariance(matrix);

        long t2 = System.currentTimeMillis();

        log.info("covariance time {} ms ...", t2 - t1);

        eigenDecomposition = new EigenDecomposition(covariance.getCovarianceMatrix());

        long t3 = System.currentTimeMillis();

        log.info("eigen decomposition time {} ms ...", t3 - t2);
    }

    public DataSet project(DataSet original, int firstMComponent) {

        long t1 = System.currentTimeMillis();

        double[][] projectArray = new double[firstMComponent][];
        IntStream.range(0, projectArray.length).parallel().forEach(i -> projectArray[i] = componentM(i, original));

        float[][] newData = new float[original.getInstanceLength()][firstMComponent];
        for (int i = 0; i < original.getInstanceLength(); i++) {
            for (int j = 0; j < firstMComponent; j++) {
                newData[i][j] = (float) projectArray[j][i];
            }
        }

        long t2 = System.currentTimeMillis();

        log.info("project time {} ms", t2 - t1);

        return new DataSet(new FullMatrix(newData, original.getFeatureMatrix().getBooleanColumnIndicator()), original.getLabels());
    }

    public double varianceExplainedM(int m, DataSet d) {
        double[] component = componentM(m, d);
        return Arrays.stream(component).map(x -> Math.pow(x, 2)).sum() / (double) d.getInstanceLength();
    }

    public double[] componentM(int m, DataSet d) {
        double[] eigenVectorM = eigenDecomposition.getV().getColumn(m);
        double[] componentM = new double[d.getInstanceLength()];
        IntStream.range(0, componentM.length).forEach(i -> componentM[i] = innerProduct(eigenVectorM, d.getInstance(i)));
        return componentM;
    }

    public double totalVar(DataSet d) {

        double totalVariance = 0;
        for (int i = 0; i < d.getInstanceLength(); i++) {
            for (int j = 0; j < d.getFeatureLength(); j++) {
                totalVariance += Math.pow(d.getEntry(i, j), 2);
            }
        }

        return totalVariance / (double) d.getInstanceLength();
    }

    private double innerProduct(double[] v1, double[] v2) {
        return IntStream.range(0, v1.length).mapToDouble(i -> v1[i] * v2[i]).sum();
    }

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

        DataSet principleTopNSet = pca.project(dataset, 50);
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
