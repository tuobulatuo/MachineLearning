package model.supervised.ecoc;

import model.supervised.svm.SVMsSMO;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 11/27/15 for machine_learning.
 */
public class ECOCSVMs extends ECOC {

    private static Logger log = LogManager.getLogger(ECOCSVMs.class);

    public static String KERNEL_CLASS_NAME = "";

    @Override
    public void configTrainable() {

        try{
            for (int i = 0; i < trainables.length; i++) {
                SVMsSMO smo = new SVMsSMO(KERNEL_CLASS_NAME);
                trainables[i] = smo;
            }
        }catch (Exception e) {
            log.error(e.getMessage(), e);
        }

    }
}
