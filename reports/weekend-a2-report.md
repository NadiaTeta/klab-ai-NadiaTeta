# Assignment 2 — Data Wrangling and Exploratory Analysis

## Question explored

This analysis explored how demographic and lifestyle factors relate to medical insurance charges. In particular, I focused on whether smoking status and region are associated with differences in insurance charges, and whether charges vary with age.

## What I found

The dataset contained 1,338 original records and 7 columns. There were no missing values, but one exact duplicate row was identified and removed, leaving 1,337 records. The original data types were already appropriate for the analysis, so no unnecessary conversions were made.

Smoking status showed a strong difference in average insurance charges. The average charge for non-smokers was approximately $8,441, while the average charge for smokers was approximately $32,050. This difference was also visible across all four regions. The highest average charge among smokers was in the southeast region, at approximately $34,845. The pivot table showed that the difference between smokers and non-smokers was consistent across regions.

I also standardized the charges using a vectorized NumPy calculation. The standardized values had a mean effectively equal to 0 and a standard deviation of 1, confirming that the computation was successful.

**Figure 1 — [Smokers have much higher average insurance charges](a2_chart1.png)**

This chart shows the large difference in average insurance charges between smokers and non-smokers.

**Figure 2 — [Insurance charges vary widely across ages](a2_chart2.png)**

This chart shows that insurance charges vary widely across ages, with some of the highest charges occurring among older customers.

## Limitation

One limitation is that this analysis identifies associations rather than proving causation. Insurance charges are affected by several factors, including BMI, age, number of children, and smoking status, so the relationship between any single variable and charges should not be interpreted as a causal effect.

## Reflection

The GroupBy and merge transformation took the longest to get right because I had to understand how the grouped statistics could be calculated separately and then merged back onto every row of the original dataset. I also had to be careful not to run the merge transformation more than once, because doing so would add duplicate columns to the DataFrame.

If I had another dataset to analyze this weekend, I would spend more time planning the questions and transformations before writing the code. I would also explore more relationships between variables before deciding which findings were strong enough to include in the final visualizations.