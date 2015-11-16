package project.datapreprocess;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.NumericalComputation;
import utils.array.ArraySumUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class ProcessPriorNoBuzz {

    private static Logger log = LogManager.getLogger(ProcessPriorNoBuzz.class);

    public static void outlierCorrect(String path, String output) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(output), 1024 * 1024 * 32);
        String line;
        int outlier = 0;
        while ((line = reader.readLine()) != null) {

                line = line.trim().replace("\"", "");

            String[] es = line.split("\t");
            //-122.422888090412 37.769287340459094 -> avg
            if (es[5].trim().startsWith("90")) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < es.length; i++) {
                    if (i == 4) sb.append("-122.422888090412\t");
                    else if (i == 5) sb.append("37.769287340459094\t");
                    else sb.append(es[i].trim() + "\t");
                }

                writer.write(sb.toString().trim() + "\n");
                outlier ++;

            } else {
                writer.write(line + '\n');
            }
        }
        writer.close();

        log.info("{} outlier correct", outlier);
    }

    public static void featureExpand(String in, String out) throws Exception{

        HashSet<String> classes = new HashSet<>();
        HashSet<String> addresses = new HashSet<>();
        uniq(in, classes, addresses);

        String[] classesArray = classes.toArray(new String[0]);
        String[] addressArray = addresses.toArray(new String[0]);

        Arrays.sort(classesArray);
        Arrays.sort(addressArray);

        double[][] table = new double[addressArray.length][classesArray.length];
        IntStream.range(0, table.length).forEach(i -> Arrays.fill(table[i], 1)); // add one laplace smoothing
        count(in, table, classesArray, addressArray, 878049);

        double[] logOddsAddress = new double[addressArray.length];
        logOddsAddress(in, addressArray, logOddsAddress);

        logOddsPosteriorNoBuzz(table, 3);

        write(in, out, table, addressArray, logOddsAddress);
    }

    public static void logOddsAddress(String path, String[] addressArray, double[] logOddsAddress) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            int addressIndex = Arrays.binarySearch(addressArray, es[3].trim());
            logOddsAddress[addressIndex] ++;
        }
        ArraySumUtil.normalize(logOddsAddress);
        IntStream.range(0, logOddsAddress.length).forEach(i ->
                logOddsAddress[i] = NumericalComputation.logOdds(logOddsAddress[i]));

        log.info("logOdds Address");
    }

    public static void uniq(String path, HashSet<String> classes, HashSet<String> addresses) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            classes.add(es[es.length - 1].trim()); // class
            addresses.add(es[3].trim());
        }
        log.info("classes count: {}", classes.size());
        log.info("addresses count: {}", addresses.size());
    }

    public static void count(String path, double[][] table, String[] classes, String[] addresses, int lineMax) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        int lineCounter = 0;
        while ((line = reader.readLine()) != null && lineCounter < lineMax) {
            String[] es = line.trim().split("\t");
            String classString = es[es.length - 1].trim();
            String address = es[3].trim();
            int classIndex = Arrays.binarySearch(classes, classString);
            int addressIndex = Arrays.binarySearch(addresses, address);
            table[addressIndex][classIndex] ++;
            lineCounter ++;
        }
        log.info("count table...");
    }

    public static void write(String pathIn, String pathOut, double[][] table, String[] addresses, double[] logOddsAddress) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(pathIn), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(pathOut), 1024 * 1024 * 32);
        String line;
        int counter = 1;
        while ((line = reader.readLine()) != null) {

            String[] es = line.split("\t");

            String timeString = time(es[0]);
            StringBuilder builder = new StringBuilder(timeString);

            builder.append("\t");
            for (int i = 1; i < es.length; i++) {

                if (i == 3) { // handle address string
                    String addressString = address(es[i].trim(), addresses, table, logOddsAddress);
                    builder.append(addressString + "\t");
                }else {
                    builder.append(es[i] + "\t");
                }
            }
            String out = builder.toString().trim();
            writer.write(out + '\n');

            if (counter ++ % 100000 == 0) {
                log.info("write {}", counter);
            }
        }
        writer.close();
    }

    public static String time(String date) {

        StringBuilder sb = new StringBuilder();
        String year = date.split(" ")[0].split("-")[0].trim();
        String month = date.split(" ")[0].split("-")[1].trim();
        String day = date.split(" ")[0].split("-")[2].trim();
        String hour = date.split(" ")[1].split(":")[0].trim();
        String dayPeriod = dayPeriod(Integer.parseInt(hour));
        String monthPeriod = monthPeriod(Integer.parseInt(month));
        sb.append(year + "\t"+ month + "\t" + day + "\t" + hour + "\t" + dayPeriod + "\t" + monthPeriod);

        return sb.toString();
    }

    public static String monthPeriod (int month) {
        if (month >= 3 && month < 6) {
            return "spring";
        } else if (month >= 6 && month < 9) {
            return "summer";
        } else if (month >= 9 && month < 12) {
            return "fall";
        } else {
            return "winter";
        }
    }

    public static String dayPeriod(int hour) {
        if (hour >= 0 && hour < 7){
            return "after-midnight";
        }else if (hour >= 7 && hour < 20) {
            return "daytime";
        }else {
            return "night";
        }
    }

    public static void logOddsPosteriorNoBuzz(double[][] table, int rareThreshold) {

        double[] backgroundProbs = new double[table[0].length];
        IntStream.range(0, table.length).forEach(i ->
                IntStream.range(0, table[i].length).forEach(j -> backgroundProbs[j] += table[i][j]));
        ArraySumUtil.normalize(backgroundProbs);

        AtomicInteger rareEvent = new AtomicInteger(0);
        IntStream.range(0, table.length).forEach(i -> {
            if (Arrays.stream(table[i]).sum() <= table[i].length + rareThreshold) {
                Arrays.fill(table[i], 1.0);
                rareEvent.getAndIncrement();
            }else {
                double[] probs = ArraySumUtil.normalize(table[i]);
                IntStream.range(0, probs.length).forEach(j -> Math.pow(probs[j] /= backgroundProbs[j], 3));
            }
        });

        log.info("logOddsPosteriorNoBuzz");
        log.info("rare {}", rareEvent.get());
        log.info("backgroundProbs {}", backgroundProbs);
    }

    public static String address(String address, String[] addresses, double[][] table, double[] logOddsAddress) {

        StringBuilder builder = new StringBuilder();

        if (address.contains("/")) {
            builder.append("intersect" + "\t");
        } else {
            builder.append("non-intersect" + "\t");
        }

        int addressIndex = Arrays.binarySearch(addresses, address);
        double logOdd = logOddsAddress[addressIndex];
        builder.append(logOdd + "\t");
        double[] logOdds = table[addressIndex];
        Arrays.stream(logOdds).forEach(x -> builder.append(x + "\t"));

        return builder.toString().trim();
    }

    public static void main(String[] args) throws Exception{

        String raw = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/posterior/data.all.raw.txt";
        String correct = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/nobuzz/data.all.correct.txt";
        String expand = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/nobuzz/data.all.expand.txt";

//        outlierCorrect(raw, correct);

        featureExpand(correct, expand);
    }
}
