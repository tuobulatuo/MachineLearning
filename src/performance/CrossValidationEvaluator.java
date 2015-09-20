package performance;

import com.google.common.primitives.Ints;
import data.DataSet;
import data.core.Norm;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import model.supervised.cart.ClassificationTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Random;
import java.util.function.IntPredicate;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class CrossValidationEvaluator {

    private static final Logger log = LogManager.getLogger(CrossValidationEvaluator.class);

    private DataSet rowDataSet = null;

//    private int k = 0;

    private int[][] kFoldIndex = null;

    private Evaluator evaluator;

    private boolean needNorm = true;

    private Norm norm;

    public CrossValidationEvaluator(Evaluator eva, DataSet dataSet, int k, Norm norm) {

        evaluator = eva;
        rowDataSet = dataSet;
        this.norm = norm;

        TIntList instancesIndex = new TIntArrayList(IntStream.range(0, rowDataSet.getInstanceLength()).toArray());
        instancesIndex.shuffle(new Random());
        kFoldIndex = new int[k][];
        int pointer = 0;
        for (int i = 0; i < k; i++) {
            int len = Math.min(rowDataSet.getInstanceLength() / k, instancesIndex.size() - pointer);
            int[] a = new int[len];
            for (int j = 0; j < len; j++) {
                a[j] = instancesIndex.get(pointer++);
            }
            kFoldIndex[i] = a;
        }
    }

    public void crossValidateEvaluate(Trainable model) {

        double avgOnTest = 0;
        double avgOnTrain = 0;

        for (int i = 0; i < kFoldIndex.length; i++) {

            TIntHashSet testIndexes = new TIntHashSet(kFoldIndex[i]);
            IntPredicate pred = (n) -> !testIndexes.contains(n);
            int[] trainIndexes = IntStream.range(0, rowDataSet.getInstanceLength()).filter(pred).toArray();

            DataSet trainSet = rowDataSet.subDataSetByRow(trainIndexes);
            DataSet testSet = rowDataSet.subDataSetByRow(testIndexes.toArray());

            if (norm == null){}
            else if (norm.equals(Norm.MEANSD)) {
                trainSet.meanVarianceNorm();
                testSet.meanVarianceNorm(trainSet.getMeanOrMin(), trainSet.getSdOrMax());
            } else if (norm.equals(Norm.MINMAX)) {
                trainSet.shiftCompressNorm();
                testSet.shiftCompressNorm(trainSet.getMeanOrMin(), trainSet.getSdOrMax());
            }

            model.initialize(trainSet);
            model.train();
            Predictable trainedModel = model.offer();

            evaluator.setTestSet(testSet);
            evaluator.getPredictLabel(trainedModel);
            double performOnTest = evaluator.evaluate();

            evaluator.setTestSet(trainSet);
            evaluator.getPredictLabel(trainedModel);
            double performOnTrain = evaluator.evaluate();


            log.info("FOLD {}/test{}/train{}", i, performOnTest, performOnTrain);

            avgOnTest += performOnTest;
            avgOnTrain += performOnTrain;
        }

        avgOnTest /= kFoldIndex.length;
        avgOnTrain /= kFoldIndex.length;
        log.info("avgOnTest {}, avgOnTrain {}", avgOnTest, avgOnTrain);
    }
}
