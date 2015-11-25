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

//	public static void reverse(long[] array) {
//		for (int i = 0; i < array.length / 2; i++) {
//			long temp = array[i];
//			array[i] = array[array.length - i - 1];
//			array[array.length - i - 1] = temp;
//		}
//	}
//
//	public static void reverse(float[] array) {
//		for (int i = 0; i < array.length / 2; i++) {
//			float temp = array[i];
//			array[i] = array[array.length - i - 1];
//			array[array.length - i - 1] = temp;
//		}
//	}
    
	public static void reverse(double[] array) {
		for (int i = 0; i < array.length / 2; i++) {
			double temp = array[i];
			array[i] = array[array.length - i - 1];
			array[array.length - i - 1] = temp;
		}
	}

//	public static String fill(String str, int length) {
//
//		String template = "";
//		for (int i = 0; i < length; i++) {
//			template += " ";
//		}
//
//		int count = 0;
//		for (char c : str.toCharArray()) {
//			if (c < 127) {
//				count++;
//			} else {
//				count += 2;
//			}
//		}
//		if (count < length)
//			return str + template.substring(count);
//		else
//			return str;
//	}

	public static double innerProduct(double[] x1, double[] x2) {

		double result = 0;
		for (int i = 0; i < x1.length; i++) {
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
		return Arrays.stream(x3).map(xi -> Math.pow(xi, 2)).sum();
	}

	public static double normL2(double[] x) {
		return Math.pow(Arrays.stream(x).map(xi -> Math.pow(xi, 2)).sum(), 0.5);
	}
}
