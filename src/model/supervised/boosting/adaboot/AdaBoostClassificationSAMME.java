package model.supervised.boosting.adaboot;

import data.DataSet;
import model.Predictable;
import model.Trainable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.array.ArraySumUtil;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;
import performance.ClassificationEvaluator;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Created by hanxuan on 10/30/15 for machine_learning.
 */
public class AdaBoostClassificationSAMME implements Trainable, Predictable{

    private static Logger log = LogManager.getLogger(AdaBoostClassificationSAMME.class);

    private DataSet trainingData = null;

    private DataSet testingData = null;

    private double[] alpha = null;

    private double[] weights = null;

    private int classCount = Integer.MIN_VALUE;

    private AdaBoostClassifier[] adaBoostClassifiers = null;

    private ClassificationEvaluator roundEvaluator = null;

    private double[] roundTrainingError = null;

    private double[] roundTestingError = null;

    private double[] roundError = null;

    private double[] roundTestingAUC = null;

    public AdaBoostClassificationSAMME(){}


    @Override
    public double predict(double[] feature) {
        double[] classScores = new double[classCount];
        for (int i = 0; i < alpha.length; i++) {
            if (alpha[i] == 0) continue;
            int predictClass = (int) adaBoostClassifiers[i].boostPredict(feature);
            classScores[predictClass] += alpha[i];
        }
        int[] indexes = RandomUtils.getIndexes(classCount);
        SortIntDoubleUtils.sort(indexes, classScores);
        return indexes[classCount - 1];
    }

    @Override
    public double score(double[] feature){
        double[] classScores = new double[classCount];
        for (int i = 0; i < alpha.length; i++) {
            if (alpha[i] == 0) continue;
            int predictClass = (int) adaBoostClassifiers[i].boostPredict(feature);
            classScores[predictClass] += alpha[i];
        }
        return classScores[1] - classScores[0];
    }

    @Override
    public void train() {

        for (int i = 0; i < adaBoostClassifiers.length; i++) {
            AdaBoostClassifier classifier = adaBoostClassifiers[i];
            classifier.boostInitialize(trainingData, weights);
            classifier.boost();

            roundEvaluator.initialize(testingData, this);
            roundEvaluator.getPredictLabel();
            roundTestingError[i] = roundEvaluator.evaluate();
            roundTestingAUC[i] = roundEvaluator.getArea();

            roundEvaluator.initialize(trainingData, this);
            roundEvaluator.getPredictLabel();
            roundTrainingError[i] = roundEvaluator.evaluate();

            double error = classifier.getWeightedError();
            roundError[i] = error;

            alpha[i] = Math.log((1 - error) / error) + Math.log(classCount - 1);
            weights = classifier.getModifiedWeights();
            ArraySumUtil.normalize(weights);
        }

        log.info("AdaBoostClassificationSAMME Training finished ...");
        log.info("roundTestingError: {}", roundTestingError);
        log.info("roundTestingAUC: {}", roundTestingAUC);
        log.info("roundTrainingError: {}", roundTrainingError);
        log.info("roundError: {}", roundError);
        log.info("alpha: {}", alpha);


        try{
            TimeUnit.SECONDS.sleep(1);
        }catch (Exception e){
            log.error(e.getMessage(), e);
        }

    }

    @Override
    public void initialize(DataSet d) {

        trainingData = d;
        classCount = trainingData.getLabels().getIndexClassMap().size();

        int instanceLength = trainingData.getInstanceLength();
        weights = new double[instanceLength];
        Arrays.fill(weights, 1 / (double) instanceLength);
    }


    public void boostConfig(int iteration, String classifierClassName, ClassificationEvaluator evaluator, DataSet testData)
    throws Exception{

        AdaBoostClassifier[] classifiers = new AdaBoostClassifier[iteration];
        for (int i = 0; i < iteration; i++) {
            classifiers[i] = (AdaBoostClassifier) Class.forName(classifierClassName).getConstructor().newInstance();
        }

        roundTestingError = new double[classifiers.length];
        roundTrainingError = new double[classifiers.length];
        roundError = new double[classifiers.length];
        roundTestingAUC = new double [classifiers.length];

        alpha = new double[classifiers.length];
        adaBoostClassifiers = classifiers;
        roundEvaluator = evaluator;
        testingData = testData;

        log.info("AdaBoostClassificationSAMME config: ");
        log.info("classifiers count: {}", classifiers.length);
        log.info("classifiers CLASS: {}", classifiers.getClass().toString());
    }
}
