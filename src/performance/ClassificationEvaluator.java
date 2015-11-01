package performance;

import data.core.Label;
import gnu.trove.list.array.TDoubleArrayList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.array.ArrayUtil;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.Arrays;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class ClassificationEvaluator extends Evaluator{

    private static final Logger log = LogManager.getLogger(ClassificationEvaluator.class);

    public static int SAMPLES = 500;

    public static int POS = 1;

    public static boolean ROC = false;

    public static boolean CONFUSION_MATRIX = false;

    public int truePos = 0;

    public int falsePos = 0;

    public int trueNeg = 0;

    public int falseNeg = 0;

    public int correct = 0;

    public double[] tpr;

    public double[] fpr;

    public double area;

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

        if (CONFUSION_MATRIX) {
            printConfusionMatrix();
        }

        if (ROC) {
            getArea();
            printROC();
        }

        return correct / (double) instanceLength;
    }

    public double getArea() {

        int instanceLength = testSet.getInstanceLength();

        double[] score = IntStream.range(0, instanceLength).mapToDouble(i -> model.score(testSet.getInstance(i))).toArray();
        int[] label = IntStream.range(0, instanceLength).map(i -> (int) testSet.getLabel(i)).toArray();

        long positiveCount = Arrays.stream(label).filter(x -> x - POS == 0).count();
        long negativeCount = instanceLength - positiveCount;

        SortIntDoubleUtils.sort(label, score);
        ArrayUtil.reverse(label);
        ArrayUtil.reverse(score);

        int sampleLength = Math.min(SAMPLES, instanceLength);
        TDoubleArrayList tprList = new TDoubleArrayList(sampleLength);
        TDoubleArrayList fprList = new TDoubleArrayList(sampleLength);
        int sampleGap = (int) Math.ceil(instanceLength / (double) sampleLength);

        log.debug("positiveCount {}  negativeCount {} sampleGap {}", positiveCount, negativeCount, sampleGap);

        int pointer = 0;
        int fp = 0;
        int tp = 0;
        while (pointer < instanceLength){

            if (label[pointer] == POS){
                ++ tp;
            }else {
                ++ fp;
            }

            if (pointer % sampleGap == 0) {
                tprList.add(tp / (double) positiveCount);
                fprList.add(fp / (double) negativeCount);
            }
            ++ pointer;
        }

        tpr = tprList.toArray();
        fpr = fprList.toArray();

        area = 0;
        for (int i = 1; i < tpr.length; i++) {
            area += (fpr[i] - fpr[i - 1]) * (tpr[i - 1] + tpr[i]);
        }
        area /= 2;

        return area;
    }

    public void printROC() {

        log.info("================= ROC curve =================");
        log.info("FP: {}", Arrays.toString(fpr));
        log.info("TP: {}", Arrays.toString(tpr));
        log.info("AUC: {}", area);
        log.info("================= ========= =================");
    }

    public void printConfusionMatrix() {
        log.info("================= confusion matrix =================");

        log.info("TruePos: {}   FalsePos: {}", truePos, falsePos);
        log.info("FalseNeg: {}  TrueNeg: {}", falseNeg, trueNeg);
        log.info("precision: {} recall: {}", truePos / (double) (truePos + falsePos),
                truePos / (double) (truePos + falseNeg));
        log.info("FP rate: {} FN rate {}", falsePos / (double) (falsePos + trueNeg), falseNeg / (double) (trueNeg + falseNeg));
        log.info("accuracy: {}", (truePos + trueNeg) / (double) (truePos + trueNeg + falsePos + falseNeg));
        log.info("error: {}", (falsePos + falseNeg) / (double) (truePos + trueNeg + falsePos + falseNeg));
        log.info("================= ================ =================");
    }
}
