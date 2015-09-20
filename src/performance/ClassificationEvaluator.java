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

//    private static final Logger log = LogManager.getLogger(ClassificationEvaluator.class);

    public ClassificationEvaluator() {}

    public ClassificationEvaluator(DataSet data) {
        super(data);
    }

    public double evaluate() {

        Label trueLabel = testSet.getLabels();
        int correct = 0;
        int instanceLength = testSet.getInstanceLength();
        for (int i = 0; i < instanceLength; i++) {
            if (trueLabel.getRow(i) == predictLabel.getRow(i)) ++ correct;
        }

        return correct * 1.0 / instanceLength;
    }
}
