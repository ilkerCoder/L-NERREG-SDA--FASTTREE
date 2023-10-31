
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.ML.Data;

namespace intro
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience;
        [LoadColumn(1)]
        public float Salary;

    }
    public class SalaryPrediction
    {

        [ColumnName("Score")]
        public float Salary;

    }
}