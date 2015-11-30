package project.datapreprocess;

import gnu.trove.map.hash.TObjectIntHashMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.NumericalComputation;
import utils.array.ArraySumUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/29/15 for machine_learning.
 */
public abstract class Process {

    private static Logger log = LogManager.getLogger(Process.class);

    protected int rareThreshold = 3;

    public abstract void tableProcess(float[][] table);

    public void outlierCorrect(String path, String output) throws Exception{

        String line;
        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(output), 1024 * 1024 * 32);
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

    public void mark(String in, String output) throws Exception{

        String line;
        BufferedReader reader = new BufferedReader(new FileReader(in), 1024 * 1024 * 64);
        TObjectIntHashMap<String> counter = new TObjectIntHashMap<>(180000);
        while ((line = reader.readLine()) != null) {
            String[] es = line.split("\t");
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < es.length - 1; i++) sb.append(es[i] + "\t");

            counter.adjustOrPutValue(sb.toString(), 1, 1);
        }
        reader.close();

        reader = new BufferedReader(new FileReader(in), 1024 * 1024 * 64);
        BufferedWriter writer = new BufferedWriter(new FileWriter(output), 1024 * 1024 * 64);
        while ((line = reader.readLine()) != null) {
            String[] es = line.split("\t");
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < es.length - 1; i++) sb.append(es[i] + "\t");

            String entry = sb.toString();

            sb.append((counter.get(entry) > 1 ? 1 : 0) + "\t");
            sb.append(es[es.length - 1]);
            writer.write(sb.toString() + "\n");
        }
        reader.close();
        writer.close();

        log.info("mark ..");
        log.info("{} duplicate feature", counter.size());
    }

    public void featureExpand(String in1, String out) throws Exception{

        HashSet<String> classes = new HashSet<>();
        HashSet<String> addresses = new HashSet<>();

        uniqClassAddress(in1, classes, addresses);

        String[] classesArray = classes.toArray(new String[0]);
        String[] addressArray = addresses.toArray(new String[0]);

        Arrays.sort(classesArray);
        Arrays.sort(addressArray);


        float[][] tableClass = new float[addressArray.length][classesArray.length];
        IntStream.range(0, tableClass.length).forEach(i -> Arrays.fill(tableClass[i], 1)); // add one laplace smoothing

        count(in1, tableClass, classesArray, addressArray, 878049);

        double[] logOddsAddress = new double[addressArray.length];
        logOddsAddress(in1, addressArray, logOddsAddress);

        tableProcess(tableClass);

        write(in1, out, tableClass, addressArray, logOddsAddress);
    }

    private void logOddsAddress(String path, String[] addressArray, double[] logOddsAddress) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 64);
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

    private int uniqClassAddress(String path, HashSet<String> classes, HashSet<String> addresses) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        int lineCounter = 0;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            classes.add(es[es.length - 1].trim()); // class
            addresses.add(es[3].trim());
            lineCounter++;
        }
        reader.close();

        log.info("classes count: {}", classes.size());
        log.info("addresses count: {}", addresses.size());
        log.info("total instance: {}", lineCounter);

        return lineCounter;
    }

    private void count(String path, float[][] table, String[] classes, String[] addresses, int lineMax) throws Exception{

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

    private void write(String pathIn, String pathOut, float[][] tableClass, String[] addresses, double[] logOddsAddress)
            throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(pathIn), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(pathOut), 1024 * 1024 * 32);
        String line;
        int counter = 0;
        while ((line = reader.readLine()) != null) {

            String[] es = line.split("\t");

            String timeString = time(es[0]);
            StringBuilder builder = new StringBuilder(timeString);
            builder.append("\t");
            for (int i = 1; i < es.length; i++) {

                if (i == 3) { // handle address string
                    String addressString = address(es[i].trim(), addresses, tableClass, logOddsAddress);
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

    private String time(String date) {

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

    private String monthPeriod (int month) {
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

    private String dayPeriod(int hour) {
        if (hour >= 0 && hour < 7){
            return "after-midnight";
        }else if (hour >= 7 && hour < 20) {
            return "daytime";
        }else {
            return "night";
        }
    }

    private String address(String address, String[] addresses, float[][] tableClass, double[] logOddsAddress) {

        StringBuilder builder = new StringBuilder();

        if (address.contains("/")) {
            builder.append("intersect" + "\t");
        } else {
            builder.append("non-intersect" + "\t");
        }

        int addressIndex = Arrays.binarySearch(addresses, address);
        double logOdd = logOddsAddress[addressIndex];
        builder.append(logOdd + "\t");

        for (int i = 0; i < tableClass[addressIndex].length; i++) builder.append(tableClass[addressIndex][i] + "\t");

        return builder.toString().trim();
    }
}
