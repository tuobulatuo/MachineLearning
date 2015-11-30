package model.supervised.svm;

import data.DataSet;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;
import model.Predictable;
import model.Trainable;
import model.supervised.svm.kernels.GaussianK;
import model.supervised.svm.kernels.Kernel;
import model.supervised.svm.kernels.LinearK;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArrayUtil;
import utils.sort.SortIntDoubleUtils;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by hanxuan on 11/25/15 for machine_learning.
 */
public class SVMsSMO implements Trainable, Predictable{

    public static double C = 0.01;
    public static double TOL = 1E-6;
    public static double EPS1 = 1E-10;
    public static double EPS2 = 1E-15;
    public static int MAX_CHANGE = 100000;
    public static int LRU_MAX_ENTRY = 10000;
    public static boolean DEBUG = false;
    public static int PRINT_GAP = 50000;


    private static Logger log = LogManager.getLogger(SVMsSMO.class);


    private float[] alphas = null;
    private float[] errorCache = null;
    private float[] kernelCache = null;
    private float[] w = null;
    private float[] sqr = null;
    private int[] y = null;

    private Map lruKernelCache = null;


    private Kernel kernel = null;


    private double b = 0;
    private double deltaB = 0;


    private DataSet data = null;
    private int instanceLength = 0;


    private int totalChange = 0;
    private int count1 = 0;
    private int count2 = 0;


    public SVMsSMO(String kernelClassName) throws Exception{
        kernel = (Kernel) Class.forName(kernelClassName).newInstance();
        log.info("SVMsSMO use kernel {}", kernelClassName);
    }

    @Override
    public void initialize(DataSet d) {

        data = d;
        instanceLength = data.getInstanceLength();
        alphas = new float[instanceLength];
        errorCache = new float[instanceLength];
        kernelCache = new float[instanceLength];
        y = new int[instanceLength];
        Arrays.fill(y, 1);

        for (int i = 0; i < instanceLength; i++) {
            if (data.getLabel(i) == 0.0D) y[i] = -1;
            double[] xi = data.getInstance(i);
            kernelCache[i] = (float) kernel.similarity(xi, xi);
            errorCache[i] = - y[i];
        }

        if (kernel.getClass().isInstance(new LinearK())) {
            w = new float[instanceLength];
        }

        if (kernel.getClass().isInstance(new GaussianK())) {
            sqr = new float[instanceLength];
            for (int i = 0; i < instanceLength; i++) {
                double[] xi = data.getInstance(i);
                sqr[i] = (float) ArrayUtil.innerProduct(xi, xi);
            }
        }

        totalChange = 0;
        b = 0;
        deltaB = 0;
        count1 = 0;
        count2 = 0;

        lruKernelCache = new LinkedHashMap(LRU_MAX_ENTRY+1, .75F, true) {
            public boolean removeEldestEntry(Map.Entry eldest) {
                return size() > LRU_MAX_ENTRY;
            }
        };

        log.info("SVM initialize finished ...");
    }

    @Override
    public double predict(double[] feature) {
        return f(feature) > 0 ? 1 : 0;
    }

    @Override
    public void train() {

        log.info("SVMsSMO training started ..");

        long tic = System.currentTimeMillis();
        loop();
        long toc = System.currentTimeMillis();

        log.info("SVMsSMO training finished, elapsed {} ms", toc - tic);

        statistic();

        lruKernelCache.clear();
    }

    private void statistic() {

        int counter1 = 0;
        int counter2 = 0;
        int counter3 = 0;
        int counter4 = 0;
        for (int i = 0; i < alphas.length; i++) {
            if (alphas[i] > 0.0F && alphas[i] < (float) C) counter1++;
            else if (alphas[i] == (float) C) counter2 ++;
            else if (alphas[i] == 0.0F) counter3 ++;
            else{
                log.warn("abnormal alpha {} -> {}", i, alphas[i]);
                counter4++;
            }

        }

        log.info("============ INFO ============");
        log.info("support vectors count {}", counter1);
        log.info("in margin count {}", counter2);
        log.info("out margin {}", counter3);
        log.info("abnormal alphas count {}", counter4);

        log.info("hit LRU {}", count2);
        log.info("miss LRU {}", count1);
        log.info("============ ==== ============");
    }

    private void loop() {

        int numChanged = 0;
        boolean examineAll = true;

        while (numChanged > 0 || examineAll) {

            numChanged = 0;

            if (examineAll) {
                for (int i = 0; i < instanceLength; i++) {
//                    numChanged += examineExample(i);
                    numChanged += examineExampleOptimized(i);
                }
            }else {
                for (int i = 0; i < instanceLength; i++) {
//                    if (alphas[i] > 0 && alphas[i] < C) numChanged += examineExample(i);
                    if (alphas[i] > 0 && alphas[i] < C) numChanged += examineExampleOptimized(i);
                }
            }

            if (examineAll) examineAll = false;
            else if (numChanged == 0) examineAll = true;

            if (totalChange > MAX_CHANGE) break;
        }
    }

//    private int examineExample(int i) {
//
//        double errorI = error(i);
//        double ri = errorI * y[i];    // ri = Ei * Yi = f(xi)*Yi - 1
//        if ((ri < -TOL && alphas[i] < C) || (ri > TOL && alphas[i] > 0)) {
//
//            // select j by argmax |Ei - Ej|
//            int bestJ = -1;
//            double maxGap = 0;
//            for (int j = 0; j < instanceLength; j++) {
//                if (alphas[j] > 0 && alphas[j] < C) {
//                    double errorJ = error(j);
//                    double gap = Math.abs(errorI - errorJ);
//                    if (gap > maxGap) {
//                        bestJ = j;
//                        maxGap = gap;
//                    }
//                }
//            }
//
//            if (bestJ >= 0 && takeStep(errorI, i, bestJ)) return 1;
//
//            // loop over all non-boundary points
//            int randIndex = new Random(System.currentTimeMillis()).nextInt(instanceLength);
//            for (int jj = randIndex; jj < instanceLength + randIndex; jj++) {
//                int j = jj % instanceLength;
//                if (alphas[j] > 0 && alphas[j] < C) if (takeStep(errorI, i, j)) return 1;
//            }
//
//            // loop over all points
//            randIndex = new Random(System.currentTimeMillis()).nextInt(instanceLength);
//            for (int jj = randIndex; jj < instanceLength + randIndex; jj++) {
//                int j = jj % instanceLength;
//                if (takeStep(errorI, i, j)) return 1;
//            }
//        }
//
//        return 0;
//    }

    private int examineExampleOptimized(int i) {

        double errorI = error(i);
        double ri = errorI * y[i];    // ri = Ei * Yi = f(xi)*Yi - 1
        if ((ri < -TOL && alphas[i] < C) || (ri > TOL && alphas[i] > 0)) {

            // select j by argmax |Ei - Ej|
            TDoubleArrayList gap = new TDoubleArrayList(500);
            TIntArrayList idx = new TIntArrayList(500);
            for (int j = 0; j < instanceLength; j++) {
                if (alphas[j] > 0 && alphas[j] < C) {
                    gap.add(Math.abs(errorI - error(j)));
                    idx.add(j);
                }
            }

            double[] gap1 = gap.toArray();
            int[] idx1 = idx.toArray();
            SortIntDoubleUtils.sort(idx1, gap1);
            ArrayUtil.reverse(idx1);
            for (int bestJ: idx1) if (takeStep(errorI, i, bestJ)) return 1;


            TIntHashSet seen = new TIntHashSet(idx);
            // loop over all points
            int randIndex = new Random(System.currentTimeMillis()).nextInt(instanceLength);
            for (int jj = randIndex; jj < instanceLength + randIndex; jj++) {
                int j = jj % instanceLength;
                if (!seen.contains(j) && takeStep(errorI, i, j)) return 1;
            }
        }

        return 0;
    }

    private boolean takeStep(double errorI, int i, int j) {

        if (i == j) return false;

        double errorJ = error(j);

        double l, h;

        if (y[i] != y[j]){
            l = Math.max(0, alphas[j] - alphas[i]);
            h = Math.min(C, alphas[j] - alphas[i] + C);
        }else {
            l = Math.max(0, alphas[j] + alphas[i] - C);
            h = Math.min(C, alphas[j] + alphas[i]);
        }

        if (l >= h) {
            log.debug("l({}) >= h({}) for {} {}, return ...", l, h, i, j);
            return false;
        }

        double kernelIJ = getKernelFromLRU(Math.max(i, j), Math.min(i, j));
        double eta = kernelCache[i] + kernelCache[j] - 2 * kernelIJ;

        double alphaJ;
        if (eta > 0) {
            alphaJ = alphas[j] + y[j] * (errorI - errorJ) / eta;
            if (alphaJ < l) alphaJ = l;
            else if (alphaJ > h) alphaJ = h;
        } else  {
            log.debug("rare case detected, eta = {}, for {} {} ", eta, i, j);

            double a1 = - eta / 2;
            double a2 = y[j] * (errorI - errorJ) + eta * alphas[j];
            double objL = a1 * l * l + a2 * l;
            double objH = a1 * h * h + a2 * h;

            if (objL - objH > EPS2) alphaJ = l;
            else if (objH - objL > EPS2) alphaJ = h;
            else alphaJ = alphas[j];
        }

        if (Math.abs(alphaJ - alphas[j]) < EPS1 * (EPS1 + alphaJ + alphas[j])) return false;

        int s = y[i] * y[j];
        double alphaI = alphas[i] + s * (alphas[j] - alphaJ);
        if (alphaI < 0 || alphaI > C) {
            log.debug("{} alphaI < 0 || alphaI > C for {} {}, alphaJ {}", alphaI, i, j, alphaJ);
        }
        if (alphaI < EPS2) {
            alphaJ += alphaI * s;
            alphaI = 0;
        }

        if (alphaI > C - EPS2) {
            alphaJ += (alphaI - C) * s;
            alphaI = C;
        }

        if (alphaJ < EPS2) alphaJ = 0;
        if (alphaJ > C - EPS2) alphaJ = C;

        updateB(kernelIJ, alphaI, alphaJ, errorI, errorJ, i, j);
        updateErrorCache(alphaI - alphas[i], alphaJ - alphas[j], i, j);
        if (kernel.getClass().isInstance(new LinearK())) {
            updateW();
        }

        alphas[i] = (float) alphaI;
        alphas[j] = (float) alphaJ;

        if (totalChange ++ % PRINT_GAP == 0) {
            log.info("total change {}", totalChange);
            if (DEBUG) checkConstraintSum();
        }

        return true;
    }

    private void updateB(double kernelIJ, double alphaI, double alphaJ, double errorI, double errorJ, int i, int j) {

        double b1, b2;

        b1 = b - errorI + (alphas[i] - alphaI) * y[i] * kernelCache[i] + (alphas[j] - alphaJ) * y[j] * kernelIJ;
        if (alphaI > 0 && alphaI < C) {
            deltaB = b1 - b;
            b = b1;
            return;
        }

        b2 = b - errorJ + (alphas[i] - alphaI) * y[i] * kernelIJ + (alphas[j] - alphaJ) * y[j] * kernelCache[j];
        if (alphaJ > 0 && alphaJ < C) {
            deltaB = b2 - b;
            b = b2;
            return;
        }

        double bb = (b1 + b2) / 2;
        deltaB = bb - b;
        b = bb;
    }

    private void updateErrorCache(double deltaAlphaI, double deltaAlphaJ, int i, int j) {
        double ti = y[i] * deltaAlphaI;
        double tj = y[j] * deltaAlphaJ;

        for (int k = 0; k < instanceLength; k++) {
            if (alphas[k] > 0 && alphas[k] < C)
                errorCache[k] += ti * getKernelFromLRU(Math.max(i, k), Math.min(i, k)) + tj * getKernelFromLRU(Math.max(j, k), Math.min(j, k)) + deltaB;
        }

        errorCache[i] = 0; // even if alphaI == 0 or alphaI == C, we can set this way.
        errorCache[j] = 0;
    }

    private void updateW() {

    }

    private double error(int i) {
        if (alphas[i] > 0 && alphas[i] < C)
            return errorCache[i];
        else
            return f(data.getInstance(i)) - y[i];
    }

    private double f(double[] x) {
        double result = 0;
        for (int i = 0; i < instanceLength; i++) {
            if (alphas[i] > 0) result += alphas[i] * y[i] * kernel.similarity(data.getInstance(i), x);
        }
        return result + b;
    }

    private double getKernelFromLRU(int i, int j) {

        String key = i + "-" + j;
        Object ans = lruKernelCache.get(key);
        if (ans == null) {
            count1 ++;
            double similarity;
            if (kernel.getClass().isInstance(new GaussianK())) {
                similarity = ((GaussianK) kernel).similarity(sqr[i], sqr[j], data.getInstance(i), data.getInstance(j));
            }else {
                similarity = kernel.similarity(data.getInstance(i), data.getInstance(j));
            }

            lruKernelCache.put(key, similarity);
            return similarity;
        }else {
            count2 ++;
            return (double) ans;
        }
    }

    private void checkConstraintSum() {
        double result = 0;
        for (int i = 0; i < instanceLength; i++) {
            if (alphas[i] > EPS2) result += alphas[i] * y[i];
        }
        log.info("check sum {}", result);
    }
}
