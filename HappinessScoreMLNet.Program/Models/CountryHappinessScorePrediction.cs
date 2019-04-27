namespace HappinessScoreMLNet.Program.Models
{
    using Microsoft.ML.Data;

    public class CountryHappinessScorePrediction
    {
        [ColumnName("Score")]
        public float HappinessScore;
    }
}
