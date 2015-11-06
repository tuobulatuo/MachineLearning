package project.datapreprocess;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.array.ArraySumUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
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

    public static void cut(double trainRatio) throws Exception{

        String input = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.full.no.x.y.txt";
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

        logOdds(table);

        write(table, addressArray);
    }

    public static void uniq(HashSet<String> classes, HashSet<String> addresses) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.txt";
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

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.txt";
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

    public static void write(double[][] table, String[] addresses) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.txt";
        String output = "/Users/hanxuan/Dropbox/neu/fall15/data mining/project/data/clean/data.full.no.x.y.txt";
        BufferedReader reader = new BufferedReader(new FileReader(path), 1024 * 1024 * 32);
        BufferedWriter writer = new BufferedWriter(new FileWriter(output), 1024 * 1024 * 32);
        String line;
        while ((line = reader.readLine()) != null) {

            String[] es = line.split("\t");

            String date = es[0];
            String month = date.split(" ")[0].split("-")[1];
            String day = date.split(" ")[0].split("-")[2];
            String hour = date.split(" ")[1].split(":")[0];

            StringBuilder builder = new StringBuilder(month + "\t" + day + "\t" +hour + "\t");
            for (int i = 1; i < es.length; i++) {
                if (i == 4 || i == 5) continue;
                if (i == 3) {
                    String address = es[i].trim();
                    int addressIndex = Arrays.binarySearch(addresses, address);
                    double[] logOdds = table[addressIndex];
                    Arrays.stream(logOdds).forEach(x -> builder.append(x + "\t"));
                }else {
                    builder.append(es[i] + "\t");
                }
            }
            String out = builder.toString().trim();
            writer.write(out + '\n');
        }
        writer.close();
    }

    public static void logOdds(double[][] table) {
        IntStream.range(0, table.length).forEach(i -> table[i] = ArraySumUtil.normalize(table[i]));
//        IntStream.range(0, table.length).forEach(i -> {
//            double[] probs = table[i];
//            IntStream.range(0, probs.length).forEach(j -> probs[j] = Math.log(probs[j]) - Math.log(1 - probs[j]));
//        });
        log.info("logOdds");
    }



    public static void main(String[] args) throws Exception{

//        clean();
//        featureExpand();

        cut(Double.parseDouble("0.01"));
    }
}
