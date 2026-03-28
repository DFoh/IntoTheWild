from analysis.util import load_cleaned_demographics_data

if __name__ == '__main__':
    df_demo = load_cleaned_demographics_data()
    print(df_demo)
    # Add BMI column
    df_demo["BMI"] = df_demo["body_mass_kg"] / (df_demo["height_cm"] / 100) ** 2
    # Print summary statistics for age, BMI, number and percentage of female (sex == "W"), and finish time in seconds
    print("\nSummary statistics:")
    print(f"Age: mean={df_demo['age'].mean():.1f}, std={df_demo['age'].std():.1f}, min={df_demo['age'].min()}, max={df_demo['age'].max()}")
    print(f"BMI: mean={df_demo['BMI'].mean():.1f}, std={df_demo['BMI'].std():.1f}, min={df_demo['BMI'].min():.1f}, max={df_demo['BMI'].max():.1f}")
    num_female = (df_demo["sex"] == "W").sum()
    print(f"Female: {num_female:.0f}")
    print(f"Female percentage: {num_female / len(df_demo) * 100:.1f}%")
    print(f"Finish time (s): mean={df_demo['finish_time_s'].mean():.1f}, std={df_demo['finish_time_s'].std():.1f}, min={df_demo['finish_time_s'].min():.1f}, max={df_demo['finish_time_s'].max():.1f}")
    finish_time_mean_minutes = df_demo['finish_time_s'].mean() // 60
    finish_time_mean_seconds = df_demo['finish_time_s'].mean() % 60
    print(f"Finish time (mean): {finish_time_mean_minutes:.0f}m {finish_time_mean_seconds:.1f}s")
    finish_time_std_minutes = df_demo['finish_time_s'].std() // 60
    finish_time_std_seconds = df_demo['finish_time_s'].std() % 60
    print(f"Finish time (std): {finish_time_std_minutes:.0f}m {finish_time_std_seconds:.1f}s")
    print(f"Finish time (mean): {finish_time_mean_minutes:.0f}m {finish_time_mean_seconds:.1f}s ± {finish_time_std_minutes:.0f}m {finish_time_std_seconds:.1f}s")

