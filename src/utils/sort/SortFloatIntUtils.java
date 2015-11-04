package utils.sort;

import java.util.Arrays;

public class SortFloatIntUtils {

	public static void sort(float[] y, int[] x) {
		sort1(x, 0, x.length, y);
	}

	private static void sort1(int x[], int off, int len, float y[]) {
		// Insertion sort on smallest arrays
		if (len < 7) {
			for (int i = off; i < len + off; i++)
				for (int j = i; j > off && x[j - 1] > x[j]; j--) {
					swap(x, j, j - 1);
					swap(y, j, j - 1);
				}
			return;
		}

		// Choose a partition element, v
		int m = off + (len >> 1); // Small arrays, middle element
		if (len > 7) {
			int l = off;
			int n = off + len - 1;
			if (len > 40) { // Big arrays, pseudomedian of 9
				int s = len / 8;
				l = med3(x, l, l + s, l + 2 * s);
				m = med3(x, m - s, m, m + s);
				n = med3(x, n - 2 * s, n - s, n);
			}
			m = med3(x, l, m, n); // Mid-size, med of 3
		}
		double v = x[m];

		// Establish Invariant: v* (<v)* (>v)* v*
		int a = off, b = a, c = off + len - 1, d = c;
		while (true) {
			while (b <= c && x[b] <= v) {
				if (x[b] == v) {
					swap(x, a, b);
					swap(y, a++, b);
				}
				b++;
			}
			while (c >= b && x[c] >= v) {
				if (x[c] == v) {
					swap(x, c, d);
					swap(y, c, d--);
				}
				c--;
			}
			if (b > c)
				break;
			swap(x, b, c);
			swap(y, b++, c--);
		}

		// Swap partition elements back to middle
		int s, n = off + len;
		s = Math.min(a - off, b - a);
		vecswap(x, off, b - s, s, y);
		s = Math.min(d - c, n - d - 1);
		vecswap(x, b, n - s, s, y);

		// Recursively sort non-partition-elements
		if ((s = b - a) > 1)
			sort1(x, off, s, y);
		if ((s = d - c) > 1)
			sort1(x, n - s, s, y);
	}

	private static void swap(float x[], int a, int b) {
		float t = x[a];
		x[a] = x[b];
		x[b] = t;
	}
	
	private static void swap(int x[], int a, int b) {
		int t = x[a];
		x[a] = x[b];
		x[b] = t;
	}

	private static int med3(int x[], int a, int b, int c) {
		return (x[a] < x[b] ? (x[b] < x[c] ? b : x[a] < x[c] ? c : a)
				: (x[b] > x[c] ? b : x[a] > x[c] ? c : a));
	}

	private static void vecswap(int x[], int a, int b, int n, float y[]) {
		for (int i = 0; i < n; i++, a++, b++) {
			swap(x, a, b);
			swap(y, a, b);
		}
	}

	public static void main(String[] args) {
		float[] users = new float[] {3, 34, 6, 1, 23, 9, 80, 34, 67, 89, 31, 45, 900};
		int[] values = new int[] {3, 34, 6, 1, 23, 9, 80, 34, 67, 89, 31, 45, 900};
		sort(users, values);
		System.out.println(Arrays.toString(users));
	}

}
