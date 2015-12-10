package model.supervised.kernels;

/**
 * Created by hanxuan on 12/10/15 for machine_learning.
 */
public class KernelMain {

    public static void main(String[] args) {

        double[] p1 = new double[]{0.5, 0.5};
        double[] p2 = new double[]{1, 1};
        double[] p3 = new double[]{1, 0};
        double[] p4 = new double[]{0, 1};

        Kernel k1 = new Cosine();
        Kernel k2 = new Euclidean();

        System.out.println(k1.similarity(p1, p2));
        System.out.println(k1.similarity(p3, p2));
        System.out.println(k1.similarity(p1, p3));
        System.out.println(k1.similarity(p1, p4));

        System.out.println(k2.similarity(p1, p2));
        System.out.println(k2.similarity(p3, p2));
        System.out.println(k2.similarity(p1, p3));
        System.out.println(k2.similarity(p1, p4));
    }
}
