package ztest;

import gnu.trove.impl.sync.TSynchronizedIntObjectMap;
import gnu.trove.map.hash.TDoubleIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Clock;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by hanxuan on 9/10/15.
 */
public class Test {

    public static void main(String[] args) {
//        System.out.println("hello");
//        Clock c = Clock.systemUTC();
//        System.out.println();
//        List<String> name = Arrays.asList("baa", "1", "2s", "a");
//        Collections.sort(name, (String a, String b) -> b.length() > a.length() ? 1: -1);
//        System.out.println(name);


//        int[][] a = new int[2][3];
//        System.out.println(a.length);
//        System.out.println(a[0].length);



//        List<String> l = Arrays.asList("1", "2");

//      System.out.print(new String[]{"1", "2", "3"}[-1]);
//        String a = "a        b";
//        System.out.println(Arrays.toString(a.split("\\s+")));


//        System.out.println(Arrays.toString(IntStream.range(0, 10).toArray()));

        Logger log = LogManager.getLogger(Test.class);
//        ExecutorService service = Executors.newFixedThreadPool(1);
//        CountDownLatch countDownLatch = new CountDownLatch(10);
////        AtomicInteger counter = new AtomicInteger(0);
//        for (int i = 0; i < 10 ; i++) {
//            service.submit(
//                    () -> {
//                        try {
//                            log.info("sleep2");
//                            TimeUnit.SECONDS.sleep(2);
//                        } catch (Throwable t) {
//
//                        }
//                        countDownLatch.countDown();
//                    });
//        }
//
//        try {
//            countDownLatch.await();
//            Thread.sleep(1000);
//        }catch (Throwable t) {
//            System.out.println(t);
//        }
//        service.shutdown();
//
//
//
//        log.info("after shutdown");
//
//        log.info("still");
//
//        return;

        int[] ids = {1,1,2,2,2,3,3,3,4,5,6,7,7,7};
        int p = 1;
        while (p < ids.length) {
            if (ids[p] == ids[p - 1]) {
                ++ p;
                continue;
            }

            log.info("{} {}", p, ids[p]);
            ++ p;
        }
    }
}
