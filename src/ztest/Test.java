package ztest;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.neu.util.rand.RandomUtils;
import org.neu.util.sort.SortIntDoubleUtils;

import java.util.Arrays;

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

//


//        double[][] m0 = new double[][] {
//                {1, 96,    26,    26,    55},
//                {1, 55,    82,    62,    92},
//                {1, 14,    25,    48,    29},
//                {1, 15,    93,    36,    76},
//                {1, 26,    35,    84,    76},
//                {1, 85,    20,    59,    39}
//
//        };

//        RealMatrix p = new Array2DRowRealMatrix(m0, false);
//
//        p = p.transpose().multiply(p);
//
//        RealMatrix pInverse = new LUDecomposition(p).getSolver().getInverse();
//
//        log.info(Arrays.deepToString(pInverse.getData()));


//        double[][] m1 = ;
//
//        log.info(m1);
//
//        m1[0][1] = 0;
//        log.info(Arrays.deepToString(m0));



        double[] a = {0.16472572793853585, 0.16473288429423177, 0.16364715196495644, 0.16387661017209518, 0.1623304921994822, 0.006160342174793988, 0.0024250452754503546, 0.1614497432746957};
        int[] idx = RandomUtils.getIndexes(a.length);
        SortIntDoubleUtils.sort(idx, a);
        System.out.println(Arrays.toString(a));
        System.out.println(Arrays.toString(idx));
    }
}
