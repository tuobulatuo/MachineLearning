package project.datapreprocess;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 11/29/15 for machine_learning.
 */
public class ProcessMain {

    private static Logger log = LogManager.getLogger(ProcessMain.class);

    public static void main(String[] args) throws Exception{

//        noBuzz();

        prior();

        log.info("pre precess finished, exit ..");
    }


    public static void noBuzz() throws Exception{

        String raw = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/raw/data.all.raw.txt";
        String correct = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/raw/data.all.correct.txt";
        String mark = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/raw/data.all.mark.txt";
        String expand = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/nobuzz/data.all.expand.txt";

        Process process = new PriorNoBuzz();
        process.outlierCorrect(raw, correct);
        process.mark(correct, mark);
        process.featureExpand(mark, expand);
    }

    public static void prior() throws Exception{

        String raw = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/raw/data.all.raw.txt";
        String correct = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/raw/data.all.correct.txt";
        String mark = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/raw/data.all.mark.txt";
        String expand = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/prior/data.all.expand.txt";

        Process process = new ProcessPrior();
        process.outlierCorrect(raw, correct);
        process.mark(correct, mark);
        process.featureExpand(mark, expand);
    }
}
