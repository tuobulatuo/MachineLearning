package project.datapreprocess;

/**
 * Created by hanxuan on 11/29/15 for machine_learning.
 */
public class ProcessMain {

    public static void main(String[] args) throws Exception{

        String raw = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/posterior/data.all.raw.txt";
        String correct = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/posterior.10/data.all.correct.txt";
        String expand = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/posterior.10/data.all.expand.txt";

        Process process = new PriorNoBuzz();
        process.outlierCorrect(raw, correct);
//        process.featureExpand(correct, expand);
    }
}
