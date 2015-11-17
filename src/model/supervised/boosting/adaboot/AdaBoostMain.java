package model.supervised.boosting.adaboot;

import data.DataSet;
import data.builder.Builder;
import data.builder.FullMatrixDataSetBuilder;
import data.builder.SparseMatrixDataSetBuilder;
import gnu.trove.set.hash.TIntHashSet;
import model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree;
import model.supervised.boosting.adaboot.adaboostclassifier.WeightedClassificationTree;
import model.supervised.cart.Tree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import performance.ClassificationEvaluator;
import performance.CrossValidationEvaluator;
import utils.random.RandomUtils;

import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/31/15 for machine_learning.
 */
public class AdaBoostMain {

    private static Logger log = LogManager.getLogger(AdaBoostMain.class);

    public static void DecisionStumpTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = true;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            samme.boostConfig(300, className, new ClassificationEvaluator(), testSet);
            samme.train();

            samme.topFeatureCalc();
            int[] topNFeature = samme.topNFeatures(15);
            log.info("topN features {}", topNFeature);

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabelByProbs();
            evaluator.getArea();
            evaluator.printROC();

            break;
        }
    }

    public static void DecisionStumpOnPollutedSetTest() throws Exception{

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

        Tree.THREAD_WORK_LOAD = 300;

        SAMME.NEED_ROUND_REPORT = true;
        int trainSize = 4140;
        int allSize = 4601;

        DataSet trainSet = dataset.subDataSetByRow(RandomUtils.getIndexes(trainSize));
        DataSet testSet = dataset.subDataSetByRow(IntStream.range(trainSize, allSize).toArray());

        SAMME samme = new SAMME();
        samme.initialize(trainSet);
        String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
        samme.boostConfig(10, className, new ClassificationEvaluator(), testSet);
        samme.train();

        samme.topFeatureCalc();
        int[] topNFeature = samme.topNFeatures(15);
        log.info("topN features {}", topNFeature);

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, samme);
        evaluator.getPredictLabel();
        evaluator.getArea();
        evaluator.printROC();

        /**
         *  alpha: [0.6789267938621621, 0.5758799486449195, 0.46752177387180377, 0.45954145919252765, 0.38172309521695363, 0.25857032749378756, 0.30306460373727623, 0.22789056394739626, 0.1964398940294384, 0.1785598318812742]
         roundTestingError: [0.2147505422993492, 0.2147505422993492, 0.12364425162689807, 0.13449023861171372, 0.08026030368763559, 0.08242950108459868, 0.08242950108459868, 0.07592190889370931, 0.06941431670281994, 0.06941431670281994]
         roundTestingAUC: [0.7179881050848823, 0.8418015676080218, 0.9104336523691384, 0.9263657489463956, 0.9526763559021653, 0.9606325574067539, 0.966245224309743, 0.9694749694749731, 0.9694355823388103, 0.972212375438185]
         roundTrainingError: [0.20458937198067628, 0.20458937198067628, 0.15024154589371985, 0.1427536231884058, 0.10265700483091789, 0.1089371980676328, 0.09589371980676331, 0.09975845410628015, 0.09202898550724636, 0.09396135265700478]
         roundError: [0.20458937198067398, 0.24016777745074003, 0.28190260920954346, 0.2851447928446733, 0.31789852902926086, 0.3735210868951161, 0.35294268284686037, 0.3879871335233632, 0.40302424300098366, 0.4116569925064149]
         topN features [52, 6, 15, 54, 51, 20, 1050, 1049, 1048, 1047, 1046, 1045, 1044, 1043, 1042]
         */

    }


    public static void RandomDecisionStumpTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/spambase/spambase.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 57;
        int n = 4601;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = true;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.RandomDecisionStump";
            samme.boostConfig(200, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabelByProbs();
            evaluator.getArea();
            evaluator.printROC();

            break;
        }
    }

    public static void SAMMETest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/letter-recognition.reformat3.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 20000;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = false;
        WeightedClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        WeightedClassificationTree.MAX_DEPTH = 100;
        WeightedClassificationTree.MAX_THREADS = 1;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.WeightedClassificationTree";
            samme.boostConfig(20, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabelByProbs();
            System.out.print(evaluator.evaluate());

            break;
        }
    }

    public static void SAMMESampleSimulationTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/letter-recognition.reformat3.data";
        String sep = ",";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 20000;
        int[] featureCategoryIndex = {};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMMESampleSimulation.NEED_ROUND_REPORT = true;
        SAMMESampleSimulation.SAMPLE_SIZE_COEF = 1;
        AdaBoostClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        AdaBoostClassificationTree.MAX_DEPTH = 100;
        AdaBoostClassificationTree.MAX_THREADS = 1;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMMESampleSimulation sammeSampleSimulation = new SAMMESampleSimulation();
            sammeSampleSimulation.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree";
            sammeSampleSimulation.boostConfig(50, className, new ClassificationEvaluator(), testSet);
            sammeSampleSimulation.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, sammeSampleSimulation);
            evaluator.getPredictLabelByProbs();
            System.out.print(evaluator.evaluate());

            break;
        }
    }

    public static void newsgroupTest() throws Exception{
        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/train.trec/feature_matrix.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 1754;
        int n = 11314;
        int[] featureCategoryIndex = {};
        boolean isClassification = true;

        Builder builder =
                new SparseMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet trainSet = builder.getDataSet();

        String path2 = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/8newsgroup/test.trec/feature_matrix.txt";
        builder =
                new SparseMatrixDataSetBuilder(path2, sep, hasHeader, needBias, m, n, featureCategoryIndex, isClassification);

        builder.build();

        DataSet testSet = builder.getDataSet();

        SAMMESampleSimulation.NEED_ROUND_REPORT = true;
        SAMMESampleSimulation.SAMPLE_SIZE_COEF = 1;
        AdaBoostClassificationTree.INFORMATION_GAIN_THRESHOLD = Integer.MIN_VALUE;
        AdaBoostClassificationTree.MAX_DEPTH = 100;
        AdaBoostClassificationTree.MAX_THREADS = 4;

        SAMMESampleSimulation sammeSampleSimulation = new SAMMESampleSimulation();
        sammeSampleSimulation.initialize(trainSet);
        String className = "model.supervised.boosting.adaboot.adaboostclassifier.AdaBoostClassificationTree";
        sammeSampleSimulation.boostConfig(100, className, new ClassificationEvaluator(), testSet);
        sammeSampleSimulation.train();

        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        evaluator.initialize(testSet, sammeSampleSimulation);
        evaluator.getPredictLabelByProbs();
        System.out.print(evaluator.evaluate());
    }

    public static void voteTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/vote/vote.data";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 434;
        int[] featureCategoryIndex = RandomUtils.getIndexes(m);
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = true;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            samme.boostConfig(50, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabelByProbs();
            evaluator.getArea();
            evaluator.printROC();

            break;
        }
    }

    public static void crxTest() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/crx/crx1.data.txt";
        String sep = "\\s+";
        boolean hasHeader = false;
        boolean needBias = false;
        int m = 16;
        int n = 100;

//                A1:	b, a.
//                A2:	continuous.
//                A3:	continuous.
//                A4:	u, y, l, t.
//                A5:	g, p, gg.
//                A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
//                A7:	v, h, bb, j, n, z, dd, ff, o.
//                A8:	continuous.
//                A9:	t, f.
//                A10:	t, f.
//                A11:	continuous.
//                A12:	t, f.
//                A13:	g, p, s.
//                A14:	continuous.
//                A15:	continuous.
//                A16: +,-         (class attribute)

        int[] featureCategoryIndex = {0, 3, 4, 5, 6, 8, 9, 11, 12};
        boolean classification = true;

        Builder builder =
                new FullMatrixDataSetBuilder(path, sep, hasHeader, needBias, m, n, featureCategoryIndex, classification);

        builder.build();

        DataSet dataset = builder.getDataSet();

        SAMME.NEED_ROUND_REPORT = true;

        int[][] kFoldIndex = CrossValidationEvaluator.partition(dataset, 10);
        for (int i = 0; i < kFoldIndex.length; i++) {
            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (x) -> !testIndexes.contains(x);
            int[] trainIndexes = IntStream.range(0, dataset.getInstanceLength()).filter(pred).toArray();
            DataSet trainSet = dataset.subDataSetByRow(trainIndexes);
            DataSet testSet = dataset.subDataSetByRow(testIndexes.toArray());

            SAMME samme = new SAMME();
            samme.initialize(trainSet);
            String className = "model.supervised.boosting.adaboot.adaboostclassifier.DecisionStump";
            samme.boostConfig(100, className, new ClassificationEvaluator(), testSet);
            samme.train();

            ClassificationEvaluator evaluator = new ClassificationEvaluator();
            evaluator.initialize(testSet, samme);
            evaluator.getPredictLabelByProbs();
            evaluator.getArea();
            evaluator.printROC();

            break;
        }
    }

    public static void main(String[] args) throws Exception{

//        DecisionStumpTest();

//        DecisionStumpOnPollutedSetTest();

//        RandomDecisionStumpTest();

//        SAMMETest();

//        SAMMESampleSimulationTest();

//        newsgroupTest();

//        voteTest();

//        crxTest();
    }
}
