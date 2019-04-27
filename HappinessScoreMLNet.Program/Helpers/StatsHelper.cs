namespace HappinessScoreMLNet.Program.Helpers
{
    using System;
    using System.Linq;

    public static class StatsHelper
    {
        /// <summary>
        /// Compute the correlation coefficient for two arrays.
        /// </summary>
        /// <remarks>
        /// Hat-tip: https://stackoverflow.com/a/17447920/489433
        /// </remarks>
        public static float ComputeCorrellationCoefficent(float[] values1, float[] values2)
        {
            if (values1.Length != values2.Length)
            {
                throw new ArgumentException("values must be the same length");
            }

            var avg1 = values1.Average();
            var avg2 = values2.Average();

            var sum1 = values1.Zip(values2, (x1, y1) => (x1 - avg1) * (y1 - avg2)).Sum();

            var sumSqr1 = values1.Sum(x => Math.Pow(x - avg1, 2.0));
            var sumSqr2 = values2.Sum(y => Math.Pow(y - avg2, 2.0));

            return sum1 / (float)Math.Sqrt(sumSqr1 * sumSqr2);
        }
    }
}
