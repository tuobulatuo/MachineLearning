package performance;

import com.google.common.util.concurrent.AtomicDouble;
import data.DataSet;
import data.core.Label;
import model.Predictable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/17/15.
 */
public class Evaluator {

    private static Logger log = LogManager.getLogger(Evaluator.class);

    protected DataSet testSet = null;

    protected Label predictLabel = null;

    protected Predictable model = null;

    public Evaluator() {}

    public void getPredictLabel() {
        long t1 = System.currentTimeMillis();
        int instanceLength = testSet.getInstanceLength();
        float[] predict = new float[instanceLength];
        IntStream.range(0, instanceLength).forEach(
                i -> predict[i] = (float) model.predict(testSet.getInstance(i))
        );
        predictLabel = new Label(predict, null);
        long t2 = System.currentTimeMillis();
        log.info("predictLabels get, elapsed {} ms", t2 - t1);
    }

    public double evaluate() {

        Label trueLabel = testSet.getLabels();
        int instanceLength = testSet.getInstanceLength();
        AtomicDouble accu = new AtomicDouble(0);
        IntStream.range(0, instanceLength).forEach(
                i -> accu.getAndAdd(Math.pow(trueLabel.getRow(i) - predictLabel.getRow(i), 2))
        );
        return accu.get() / (double) instanceLength;
    }

    public void initialize(DataSet testSet, Predictable model) {
        this.model = model;
        this.testSet = testSet;
    }



}
