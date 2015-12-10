package utils.array;


import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;

public class ArrayUtil {

	private static Logger log = LogManager.getLogger(ArrayUtil.class);

	public static void reverse(int[] array) {
		for (int i = 0; i < array.length / 2; i++) {
			int temp = array[i];
			array[i] = array[array.length - i - 1];
			array[array.length - i - 1] = temp;
		}
	}
    
	public static void reverse(double[] array) {
		for (int i = 0; i < array.length / 2; i++) {
			double temp = array[i];
			array[i] = array[array.length - i - 1];
			array[array.length - i - 1] = temp;
		}
	}

	public static double innerProduct(double[] x1, double[] x2) {

		double result = 0;
		for (int i = 0; i < x1.length; i++) {
			if (x1[i] == 0.0D || x2[i] == 0.0D) continue;
			result += x1[i] * x2[i];
		}

		return result;
	}

	public static double[] arraySubtract(double[] x1, double[] x2) {

		double[] result = new double[x1.length];
		for (int i = 0; i < x1.length; i++) {
			result[i] = x1[i] - x2[i];
		}
		return result;
	}

	public static double euclidean(double[] x1, double[] x2) {
		double[] x3 = arraySubtract(x1, x2);
		double result = 0;
		for (double x: x3) if (x != 0) result += x * x;
		return Math.pow(result, 0.5);
	}


	public static double KLDivergence(float[] p, float[] q) {
		double result = 0;
		for (int i = 0; i < p.length; i ++)
			if (p[i] > 0) result += p[i] * Math.log(p[i] / q[i]);
		return result;
	}

	public static double norm2(double[] x) {
		double result = innerProduct(x, x);
		return Math.pow(result, 0.5);
	}
}
