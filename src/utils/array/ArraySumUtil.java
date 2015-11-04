package utils.array;

public class ArraySumUtil {

	public static double sum(double[] items) {
		double total = 0;
		for (double item : items) {
			total += item;
		}
		return total;
	}
	
	public static long sum(long[] items) {
		long total = 0;
		for (long item : items) {
			total += item;
		}
		return total;
	}

	public static float sum(float[] items) {
		float total = 0;
		for (float item : items) {
			total += item;
		}
		return total;
	}

	public static long sum(int[] items) {
		long total = 0;
		for (int item : items) {
			total += item;
		}
		return total;
	}
	
	public static float[] normalize(float[] values) {
		float sum = sum(values);
		for (int i = 0; i < values.length; i++) {
			values[i] = values[i]/sum;
		}
		return values;
	}
	
	public static double[] normalize(double[] array){
		double sum = sum(array);
		for(int i = 0; i < array.length; i++){
			array[i] = array[i]/sum;
		}
		return array;
	}
}
