package performance;

import data.DataSet;
import data.core.Label;
import gnu.trove.list.array.TIntArrayList;
import model.Predictable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class ClassificationEvaluator extends Evaluator{

    private static final Logger log = LogManager.getLogger(ClassificationEvaluator.class);

    public static boolean ROC = false;

    public static double EPSILON = 0.45;

    private int truePos = 0;

    private int falsePos = 0;

    private int trueNeg = 0;

    private int falseNeg = 0;

    private int correct = 0;

    TIntArrayList truePositiveList;

    TIntArrayList falsePositiveList;

    public ClassificationEvaluator() {}

    public double evaluate() {

        correct = truePos = trueNeg = falsePos = falseNeg = 0;

        Label trueLabel = testSet.getLabels();
        int instanceLength = testSet.getInstanceLength();
        for (int i = 0; i < instanceLength; i++) {

            double trueValue = trueLabel.getRow(i);
            double predictValue = predictLabel.getRow(i);

            if (trueValue == predictValue) ++ correct;
            if (trueValue > 0 && predictValue > 0) ++ truePos;
            if (trueValue <=0 && predictValue > 0) ++ falsePos;
            if (trueValue > 0 && predictValue <= 0) ++ falseNeg;
            if (trueValue <= 0 && predictValue <= 0) ++ trueNeg;
        }

        printConfusionMatrix();

        if (ROC) {
            printROC();
        }

        return correct / (double) instanceLength;
    }

    private void getScore() {

        int instanceLength = testSet.getInstanceLength();
        double[] score = new double[instanceLength];

        IntStream.range(0, instanceLength).forEach(
                i -> score[i] = model.score(testSet.getInstance(i))
        );
        int[] index = RandomUtils.getIndexes(score.length);
        SortIntDoubleUtils.sort(index, score);

        truePositiveList = new TIntArrayList();
        falsePositiveList = new TIntArrayList();

        int pointer = score.length - 1;
        int tp = 0;
        int fp = 0;
        while (score[pointer] > EPSILON) {
            if (testSet.getLabel(index[pointer]) == 1) ++ tp;
            if (testSet.getLabel(index[pointer]) != 1) ++ fp;
            truePositiveList.add(tp);
            falsePositiveList.add(fp);
            -- pointer;
        }
//        log.info("score: {}", Arrays.toString(score));
    }

    public void printROC() {

        getScore();

        int[] tp = truePositiveList.toArray();
        int[] fp = falsePositiveList.toArray();

        truePositiveList = new TIntArrayList();
        falsePositiveList = new TIntArrayList();

        IntStream.range(1, fp.length).forEach(i -> {
                    if (fp[i - 1] != fp[i]) {
                        falsePositiveList.add(fp[i]);
                        truePositiveList.add(tp[i]);
                    }
                }
        );

        log.info("================= ROC curve =================");
        log.info("TP: {}", Arrays.toString(truePositiveList.toArray()));
        log.info("FP: {}", Arrays.toString(falsePositiveList.toArray()));
        log.info("================= ========= =================");
    }

    public void printConfusionMatrix() {
        log.info("================= confusion matrix =================");

        log.info("TruePos: {}   FalsePos: {}", truePos, falsePos);
        log.info("FalseNeg: {}  TrueNeg: {}", falseNeg, trueNeg);
        log.info("precision: {} recall: {}", truePos / (double) (truePos + falsePos),
                truePos / (double) (truePos + falseNeg));
        log.info("accuracy: {}", (truePos + trueNeg) / (double) (truePos + trueNeg + falsePos + falseNeg));
        log.info("================= ================ =================");
    }
}
