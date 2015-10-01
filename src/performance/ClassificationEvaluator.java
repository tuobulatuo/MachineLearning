package performance;

import data.DataSet;
import data.core.Label;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/20/15 for machine_learning.
 */
public class ClassificationEvaluator extends Evaluator{

    private static final Logger log = LogManager.getLogger(ClassificationEvaluator.class);

    private int truePos = 0;

    private int falsePos = 0;

    private int trueNeg = 0;

    private int falseNeg = 0;

    public ClassificationEvaluator() {}

    public double evaluate() {

        truePos = trueNeg = falsePos = falseNeg = 0;

        Label trueLabel = testSet.getLabels();

        int instanceLength = testSet.getInstanceLength();
        for (int i = 0; i < instanceLength; i++) {
            double trueValue = trueLabel.getRow(i);
            double predictValue = predictLabel.getRow(i);
            if (trueValue == 1 && predictValue == 1) ++ truePos;
            if (trueValue == 0 && predictValue == 1) ++ falsePos;
            if (trueValue == 1 && predictValue == 0) ++ falseNeg;
            if (trueValue == 0 && predictValue == 0) ++ trueNeg;
        }

        printConfusionMatrix();

        return (truePos + trueNeg) / (double) instanceLength;
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
