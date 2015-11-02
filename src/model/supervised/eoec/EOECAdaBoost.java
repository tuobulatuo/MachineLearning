package model.supervised.eoec;

import model.supervised.boosting.adaboot.SAMME;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public class EOECAdaBoost extends EOEC{

    private static Logger log = LogManager.getLogger(EOECAdaBoost.class);

    public static int MAX_ITERATION = 200;

    public static String ADABOOST_CLASSIFIER_CLASS_NAME = "";

    @Override
    public void configTrainable() {
        SAMME.NEED_ROUND_REPORT = false;
        try {
            for (int i = 0; i < trainables.length; i++) {
                SAMME samme = new SAMME();
                samme.boostConfig(MAX_ITERATION, ADABOOST_CLASSIFIER_CLASS_NAME, null, null);
                trainables[i] = samme;
            }
        }catch (Exception e) {
            log.error(e.getMessage(), e);
        }
    }
}
