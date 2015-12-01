package model.supervised.svm;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.builder.SparseMatrixDataSetBuilder;
import data.core.Norm;
import model.supervised.svm.kernels.GaussianK;
import model.supervised.svm.kernels.PolynomialK;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;


/**
 * Created by hanxuan on 11/26/15 for machine_learning.
 */
public class SVMMain {

    private static Logger log = LogManager.getLogger(SVMMain.class);

    public static void spamTest (String kernelClassName) throws Exception {

        String path1 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path1, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        ClassificationEvaluator eva = new ClassificationEvaluator();
        ClassificationEvaluator.ROC = false;
        ClassificationEvaluator.CONFUSION_MATRIX = true;
        CrossValidationEvaluator crossEvaluator = new CrossValidationEvaluator(eva, dataset, 10, Norm.MEANSD);
        SVMsSMO smo = new SVMsSMO(kernelClassName);
        crossEvaluator.crossValidateEvaluate(smo);
    }

    public static void main(String[] args) throws Exception{



        String linearKernelClassName = "model.supervised.svm.kernels.LinearK";
        String polynomialKernelClassName = "model.supervised.svm.kernels.PolynomialK";
        String gaussianKernelClassName = "model.supervised.svm.kernels.GaussianK";

        SVMsSMO.C = 0.05;
//        SVMsSMO.MAX_CHANGE = 100000;
        SVMsSMO.MAX_CHANGE = Integer.MAX_VALUE;
        SVMsSMO.LRU_MAX_ENTRY = 200;
        SVMsSMO.DEBUG = false;
        SVMsSMO.PRINT_GAP = 20000;
        SVMsSMO.TOL = 0.1;
        SVMsSMO.EPS1 = 0.001;
        SVMsSMO.EPS2 = 1E-8;
        spamTest(linearKernelClassName);    //AVG_TEST (0.9243478260869565), AVG_TRAIN (0.9304998792562185)



        SVMsSMO.C = 1;
        SVMsSMO.MAX_CHANGE = 100000;
//        SVMsSMO.MAX_CHANGE = Integer.MAX_VALUE;
        SVMsSMO.LRU_MAX_ENTRY = 1000000;
        SVMsSMO.DEBUG = false;
        SVMsSMO.PRINT_GAP = 10000;
        SVMsSMO.TOL = 0.01;
        SVMsSMO.EPS1 = 0.001;
        SVMsSMO.EPS2 = 1E-8;
        PolynomialK.GAMMA = 0.1;
        PolynomialK.DEGREE = 2;
        spamTest(polynomialKernelClassName);    //AVG_TEST (0.9199999999999999), AVG_TRAIN (0.93477420912823)


        SVMsSMO.C = 10;
//        SVMsSMO.MAX_CHANGE = 60000;
        SVMsSMO.MAX_CHANGE = Integer.MAX_VALUE;
        SVMsSMO.LRU_MAX_ENTRY = 1000000;
        SVMsSMO.PRINT_GAP = 3000;
        SVMsSMO.TOL = 0.01;
        SVMsSMO.EPS1 = 0.001;
        SVMsSMO.EPS2 = 1E-8;
        GaussianK.GAMMA = 1 / (double) 57;
        spamTest(gaussianKernelClassName);  //AVG_TEST (0.9354347826086957), AVG_TRAIN (0.9631731465829511)
    }

}
