package project.datapreprocess;

import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class Process {

    private static Logger log = LogManager.getLogger(Process.class);

    public static void clean() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.txt";
        String output = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data.hour.txt";
        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(output), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.split("\t");
            String data = es[0];
            String hour = data.split(" ")[1].split(":")[0];
            StringBuilder builder = new StringBuilder(hour);
            builder.append("\t");
            for (int i = 1; i < es.length; i++) {
                if (i == 3) continue;
                builder.append(es[i] + "\t");
            }
            String out = builder.toString().trim();
            writer.write(out+'\n');
        }
        writer.close();
    }

    public static void delete() throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.txt";
        String output = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.delete.txt";
        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(output), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            if (es[5].trim().startsWith("90")) continue;
            String newLine = line.replace("\"", "");
            writer.write(newLine + '\n');
        }
        writer.close();

        log.info("delete");
    }

    public static void cut(double trainRatio) throws Exception{

        String input = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.full.txt";
        String outputTrain = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/1/data.full.no.x.y.train.txt";
        String outputTest = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/1/data.full.no.x.y.test.txt";
        BufferedReader reader = new BufferedReader(new FileReader(input), 1024 * 1024 * 32);
        BufferedWriter writerTrain = new BufferedWriter(new FileWriter(outputTrain), 1024 * 1024 * 32);
        BufferedWriter writerTest = new BufferedWriter(new FileWriter(outputTest), 1024 * 1024 * 32);
        Random rand = new Random();
        String line;
        while ((line = reader.readLine()) != null) {

            if (rand.nextDouble() < trainRatio) {
                writerTrain.write(line + '\n');
            }else {
                writerTest.write(line + '\n');
            }

        }
        writerTrain.close();
        writerTest.close();
    }

    public static void featureExpand() throws Exception{

        HashSet<String> classes = new HashSet<>();
        HashSet<String> addresses = new HashSet<>();
        uniq(classes, addresses);

        String[] classesArray = classes.toArray(new String[0]);
        String[] addressArray = addresses.toArray(new String[0]);

        Arrays.sort(classesArray);
        Arrays.sort(addressArray);

        double[][] table = new double[addressArray.length][classesArray.length];
        count(table, classesArray, addressArray);

        double[] intensity = new double[addressArray.length];
        intensity(table, addressArray, intensity);

        double[] logOddsAddress = new double[addressArray.length];
        logOddsAddress(table, logOddsAddress);

        logOdds(table);

        String trainIn = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.delete.txt";
        String trainOut = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.expand.txt";
//        write(trainIn, trainOut, table, addressArray, intensity, logOddsAddress);


        String testIn = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.txt";
        String testOut = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.test.expand.txt";
        write(testIn, testOut, table, addressArray, intensity, logOddsAddress);
    }

    public static void uniq(HashSet<String> classes, HashSet<String> addresses) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.delete.txt";
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

    public static void count(double[][] table, String[] classes, String[] addresses) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.delete.txt";
        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] es = line.trim().split("\t");
            String classString = es[es.length - 1].trim();
            String address = es[3].trim();
            int classIndex = Arrays.binarySearch(classes, classString);
            int addressIndex = Arrays.binarySearch(addresses, address);
            table[addressIndex][classIndex] ++;
        }
        log.info("count table...");
    }

    public static void write(String pathIn, String pathOut, double[][] table, String[] addresses, double[] intensity, double[] logOddsAddress) throws Exception{


        double defaultIntensity = median(intensity);
        double defaultLogOddsAddress = median(logOddsAddress);
        double[] defaultLogOddsAddressClass = getTableMedian(table);

        StringBuilder defaultBuilder = new StringBuilder();
        defaultBuilder.append(defaultLogOddsAddress + "\t" + defaultIntensity + "\t");
        Arrays.stream(defaultLogOddsAddressClass).forEach(x -> defaultBuilder.append(x + "\t"));
        String defaultExpandString = defaultBuilder.toString();

        BufferedReader reader = new BufferedReader(new FileReader(pathIn), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(pathOut), 1024 * 1024 * 32);
        String line;
        int counter = 1;
        int missAddress = 0;
        while ((line = reader.readLine()) != null) {

            String[] es = line.split("\t");

            String timeString = time(es[0]);
            StringBuilder builder = new StringBuilder(timeString);

            builder.append("\t");
            for (int i = 1; i < es.length; i++) {

                if (i == 3) {

                    String address = es[i].trim();

                    if (address.contains("/")) {
                        builder.append("intersect" + "\t");
                    } else {
                        builder.append("non-intersect" + "\t");
                    }

                    int addressIndex = Arrays.binarySearch(addresses, address);
                    if (addressIndex >= 0){
                        double intensityAddress = intensity[addressIndex];
                        double logOdd = logOddsAddress[addressIndex];
                        builder.append(logOdd + "\t" + intensityAddress + "\t");
                        double[] logOdds = table[addressIndex];
                        Arrays.stream(logOdds).forEach(x -> builder.append(x + "\t"));
                    }else {
                        builder.append(defaultExpandString);
                        missAddress ++;
                    }

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
        log.info("missing address {}", missAddress);
    }

    public static double[] getTableMedian(double[][] table) {

        double[] ans = new double[table[0].length];
        IntStream.range(0, ans.length).forEach(i -> {
            double[] a = IntStream.range(0, table.length).mapToDouble(j -> table[j][i]).toArray();
            ans[i] = median(a);
        });

        return ans;
    }

    public static double median(double[] a) {
        Percentile percentile = new Percentile();
        return percentile.evaluate(a, 50);
    }

    public static String time(String date) {

        StringBuilder sb = new StringBuilder();
        String month = date.split(" ")[0].split("-")[1].trim();
        String day = date.split(" ")[0].split("-")[2].trim();
        String hour = date.split(" ")[1].split(":")[0].trim();
        String dayPeriod = dayPeriod(Integer.parseInt(hour));
        String monthPeriod = monthPeriod(Integer.parseInt(month));
        sb.append(month + "\t" + day + "\t" + hour + "\t" + dayPeriod + "\t" + monthPeriod);

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

    public static void logOdds(double[][] table) {
        IntStream.range(0, table.length).forEach(i -> table[i] = ArraySumUtil.normalize(table[i]));
        IntStream.range(0, table.length).forEach(i -> {
            double[] probs = table[i];
            IntStream.range(0, probs.length).forEach(j ->
            {
                if (probs[j] == 0) {
                    probs[j] = -10;
                } else if (probs[j] == 1) {
                    probs[j] = 10;
                } else {
                    probs[j] = Math.log(probs[j]) - Math.log(1 - probs[j]);
                }

            });
        });
        log.info("logOdds");
    }

    public static void logOddsAddress(double[][] table, double[] logOddsAddress) {

        double sum = IntStream.range(0, table.length).mapToDouble(i -> Arrays.stream(table[i]).sum()).sum();

        log.info("sum {}",sum);

        IntStream.range(0, logOddsAddress.length).forEach(i -> logOddsAddress[i] = Arrays.stream(table[i]).sum() / sum);

        log.info("probs sum {}", Arrays.stream(logOddsAddress).sum());

        IntStream.range(0, logOddsAddress.length).forEach(i -> logOddsAddress[i] = Math.log(logOddsAddress[i] / (1.0 - logOddsAddress[i])));

        log.info("logOdds address");
    }

    public static void intensity(double[][] table, String[] addressArray, double[] intensityArray) {

        TObjectDoubleMap<String> intensity = new TObjectDoubleHashMap<>(addressArray.length);
        for (int i = 0; i < addressArray.length; i++) {
            String address = addressArray[i];
            double sum = Arrays.stream(table[i]).sum();

//            if(address.contains("800 Block of BRYANT ST")) log.info("{} {}", address, sum);

            if (address.contains("/")){
                String[] as = address.split("/");
                intensity.adjustOrPutValue(as[0].trim(), sum, sum);

                try {
                    intensity.adjustOrPutValue(as[1].trim(), sum, sum);
                }catch (Exception e) {
                    log.info("address exception {}", address);
                }

            }else {
                intensity.adjustOrPutValue(address, sum, sum);
            }
        }

        for (int i = 0; i < addressArray.length; i++) {
            String address = addressArray[i];

//            if(address.contains("800 Block of BRYANT ST")) log.info("{} {}", address, sum);

            if (address.contains("/")){
                String[] as = address.split("/");
                intensityArray[i] += intensity.get(as[0].trim());
                try {
                    intensityArray[i] += intensity.get(as[1].trim());
                }catch (Exception e) {
                    log.info("address exception {}", address);
                }

            }else {
                intensityArray[i] = intensity.get(address);
            }
        }

//        int[] index = RandomUtils.getIndexes(intensityArray.length);
//        SortIntDoubleUtils.sort(index, intensityArray);
//        int count = 0;
//        for (int i = index.length - 1; i >= 0 ; i--) {
//            log.info("{} {}", addressArray[index[i]], intensityArray[i]);
//            if (count++ == 10){
//                break;
//            }
//        }
//        System.exit(0);
    }


    public static void main(String[] args) throws Exception{

//        delete(); // delete noise

//        clean();
//
        featureExpand();
//
//        cut(Double.parseDouble("0.02"));

//        System.out.println(Arrays.toString("OAK ST / LAGUNA ST".split("/")));
    }
}
