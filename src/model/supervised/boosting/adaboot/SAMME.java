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
public class SAMME implements Trainable, Predictable{

    public static boolean NEED_ROUND_REPORT = false;

    private static Logger log = LogManager.getLogger(SAMME.class);

    protected DataSet trainingData = null;

    protected DataSet testingData = null;

    protected double[] alpha = null;

    protected double[] weights = null;

    protected int classCount = Integer.MIN_VALUE;

    protected AdaBoostClassifier[] adaBoostClassifiers = null;

    protected ClassificationEvaluator roundEvaluator = null;

    protected double[] roundTrainingError = null;

    protected double[] roundTestingError = null;

    protected double[] roundError = null;

    protected double[] roundTestingAUC = null;

    public SAMME(){}

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

            double error = getWeightedError(classifier);

            alpha[i] = Math.log((1 - error) / error) / 2 + Math.log(classCount - 1);

            modifyWeights(classifier, alpha[i]);
            ArraySumUtil.normalize(weights);

            if (NEED_ROUND_REPORT) {
                statisticReport(i, error);
            }

            log.info("{} round boosting finished ...", i);
        }

        log.info("SAMME Training finished ...");

        if (NEED_ROUND_REPORT) {
            printRoundReport();
        }

        try{
            TimeUnit.SECONDS.sleep(1);
        }catch (Exception e){
            log.error(e.getMessage(), e);
        }
    }

    protected void printRoundReport() {

        log.info("======================= Round Report =======================");
        log.info("alpha: {}", alpha);
        log.info("roundTestingError: {}", roundTestingError);
        log.info("roundTestingAUC: {}", roundTestingAUC);
        log.info("roundTrainingError: {}", roundTrainingError);
        log.info("roundError: {}", roundError);
        log.info("======================= ============ =======================");
    }

    protected void statisticReport(int round, double error){

        roundError[round] = error;

        roundEvaluator.initialize(testingData, this);
        roundEvaluator.getPredictLabel();
        roundTestingError[round] = 1 - roundEvaluator.evaluate();
        roundTestingAUC[round] = roundEvaluator.getArea();

        roundEvaluator.initialize(trainingData, this);
        roundEvaluator.getPredictLabel();
        roundTrainingError[round] = 1 - roundEvaluator.evaluate();
    }

    protected void modifyWeights(AdaBoostClassifier classifier, double alpha){

        for (int i = 0; i < weights.length; i++) {
            double[] feature = trainingData.getInstance(i);
            if (classifier.boostPredict(feature) != trainingData.getLabel(i)){
                weights[i] *= Math.exp(alpha);
            }else {
                weights[i] *= Math.exp(-alpha);
            }
        }
    }

    protected double getWeightedError(AdaBoostClassifier classifier) {

        double weightedError = 0;
        int instanceLength = trainingData.getInstanceLength();
        for (int i = 0; i < instanceLength; i++) {
            double[] feature = trainingData.getInstance(i);
            if (classifier.boostPredict(feature) != trainingData.getLabel(i)){
                weightedError += weights[i];
            }
        }
        return weightedError;
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

        log.info("SAMME configTrainable: ");
        log.info("classifiers count: {}", classifiers.length);
        log.info("classifiers CLASS: {}", classifiers.getClass().toString());
    }
}
