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

    public void featureExpand(String in1, String in2, String out) throws Exception{

        HashSet<String> classes = new HashSet<>();
        HashSet<String> addresses = new HashSet<>();
        HashSet<String> resos = new HashSet<>();
        HashSet<String> descs = new HashSet<>();
        uniqClassAddress(in1, classes, addresses);
        uniqResoDesc(in2, resos, descs);

        String[] classesArray = classes.toArray(new String[0]);
        String[] addressArray = addresses.toArray(new String[0]);
        String[] resosArray = resos.toArray(new String[0]);
        String[] descsArray = descs.toArray(new String[0]);

        Arrays.sort(classesArray);
        Arrays.sort(addressArray);
        Arrays.sort(resosArray);
        Arrays.sort(descsArray);


        float[][] tableClass = new float[addressArray.length][classesArray.length];
        float[][] tableReso = new float[addressArray.length][resosArray.length];
        float[][] tableDesc = new float[addressArray.length][descsArray.length];
        IntStream.range(0, tableClass.length).forEach(i -> Arrays.fill(tableClass[i], 1)); // add one laplace smoothing
        IntStream.range(0, tableReso.length).forEach(i -> Arrays.fill(tableReso[i], 1)); // add one laplace smoothing
        IntStream.range(0, tableDesc.length).forEach(i -> Arrays.fill(tableDesc[i], 1)); // add one laplace smoothing

        countClass(in1, tableClass, classesArray, addressArray, 878049);
        countDescriptionResolution(in2, addressArray, resosArray, descsArray, tableReso, tableDesc);

        double[] logOddsAddress = new double[addressArray.length];
        logOddsAddress(in1, addressArray, logOddsAddress);

        tableProcess(tableClass);
        tableProcess(tableReso);
        tableProcess(tableDesc);

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

    private void uniqClassAddress(String path, HashSet<String> classes, HashSet<String> addresses) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            classes.add(es[es.length - 1].trim()); // class
            addresses.add(es[3].trim());
        }
        log.info("classes countClass: {}", classes.size());
        log.info("addresses countClass: {}", addresses.size());
    }

    private void uniqResoDesc(String path, HashSet<String> resos, HashSet<String> descs) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            resos.add(es[1].trim()); // class
            descs.add(es[2].trim());
        }
        log.info("resos countClass: {}", resos.size());
        log.info("descs countClass: {}", descs.size());
    }

    private void countClass(String path, float[][] table, String[] classes, String[] addresses, int lineMax) throws Exception{

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
        log.info("countClass table...");
    }

    private void countDescriptionResolution(String path, String[] addresses, String[] resos, String[] descs,
                                            float[][] resoTable, float[][] descTable) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {

            String[] es = line.trim().split("\t");
            String address = es[0].trim();
            String reso = es[1].trim();
            String desc = es[2].trim();
            int addressIndex = Arrays.binarySearch(addresses, address);
            int resoIndex = Arrays.binarySearch(resos, reso);
            int descIndex = Arrays.binarySearch(descs, desc);

            resoTable[addressIndex][resoIndex] ++;
            descTable[addressIndex][descIndex] ++;
        }
        log.info("countClass table...");
    }

    private void write(String pathIn, String pathOut, float[][] table, String[] addresses, double[] logOddsAddress)
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

    private String address(String address, String[] addresses, float[][] table, double[] logOddsAddress) {

        StringBuilder builder = new StringBuilder();

        if (address.contains("/")) {
            builder.append("intersect" + "\t");
        } else {
            builder.append("non-intersect" + "\t");
        }

        int addressIndex = Arrays.binarySearch(addresses, address);
        double logOdd = logOddsAddress[addressIndex];
        builder.append(logOdd + "\t");
        float[] logOdds = table[addressIndex];
        for (int i = 0; i < logOdds.length; i++) builder.append(logOdds[i] + "\t");

        return builder.toString().trim();
    }
}
