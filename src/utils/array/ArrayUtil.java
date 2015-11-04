package utils.array;


public class ArrayUtil {

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
}
