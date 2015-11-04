package project.datapreprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

/**
 * Created by hanxuan on 11/3/15 for machine_learning.
 */
public class Preprocess {

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

    public static void main(String[] args) throws Exception{
        clean();
    }
}
