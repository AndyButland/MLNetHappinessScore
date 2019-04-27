namespace HappinessScoreMLNet.Program.Models
{
    using Microsoft.ML.Data;

    public class CountryData
    {
        [LoadColumn(0)]
        public string Code;

        [LoadColumn(1)]
        public string Name;

        [LoadColumn(3)]
        public float HappinessScore;

        [LoadColumn(4)]
        public float Population;

        [LoadColumn(5)]
        public float Area;

        [LoadColumn(6)]
        public float PopulationDensity;

        [LoadColumn(7)]
        public float Coastline;

        [LoadColumn(8)]
        public float NetMigration;

        [LoadColumn(9)]
        public float InfantMortality;

        [LoadColumn(10)]
        public float GDP;

        [LoadColumn(11)]
        public float Literacy;

        [LoadColumn(12)]
        public float Phones;

        [LoadColumn(13)]
        public float Arable;

        [LoadColumn(16)]
        public float Climate;

        [LoadColumn(17)]
        public float Birthrate;

        [LoadColumn(18)]
        public float Deathrate;
    }
}
