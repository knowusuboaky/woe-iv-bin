# WoE-IV-Bin Toolkit

## Overview
The WoE-IV-Bin Toolkit is a comprehensive Python library designed to streamline the analysis and optimization of categorical variables through the calculation of Weight of Evidence (WoE) and Information Value (IV), along with enhanced binning strategies for continuous features. This toolkit empowers data scientists and analysts to uncover valuable insights, optimize feature engineering, and improve predictive modeling accuracy.

## Concept 
**Information Value (IV)**
`IV` refers to a measure that quantifies the predictive power of an individual feature (independent variable) in relation to the target variable. It is calculated by taking the sum of the products of the differences in the proportion of goods and bads (or events and non-events) and the `Weight of Evidence (WoE)` for each category or bin of a feature.

`IV` can be used to rank features in terms of their importance or strength of association with the outcome variable. The value of IV helps in determining how well a feature is able to distinguish between the target variable's classes (for example, default vs. non-default in credit scoring).

The `IV` of a feature is a single scalar value. The interpretation of IV values is often guided by `rules of thumb`, indicating the `predictive strength of the feature`:
- IV < 0.02: **Not useful for prediction**
- 0.02 ≤ IV < 0.1: **Weak predictive power**
- 0.1 ≤ IV < 0.3: **Medium predictive power**
- 0.3 ≤ IV: **Strong predictive power**

## Features
-   **WoE and IV Calculation**: Effortlessly compute WoE and IV for categorical variables, enabling deeper understanding of the predictive strength of each category within the variable.
-   **Continuous Binning Optimization**: Dynamically optimize binning strategies for continuous features by maximizing IV, facilitating superior feature engineering and model performance.
-   **Flexible and Intuitive**: Seamlessly integrate the toolkit into existing workflows, with intuitive functions for easy implementation and customization according to specific analysis requirements.
-   **Scalable Algorithms**: Utilize scalable algorithms and optimized code to handle large datasets efficiently, ensuring fast computation even with extensive data volumes.
-   **Empirical Insights**: Gain actionable insights into dataset characteristics and predictive patterns through comprehensive WoE, IV, and binning analysis.

## Workflow

<img src="https://github.com/knowusuboaky/woe-iv-bin/blob/main/README_files/figure-markdown/mermaid-figure-1.png" width="730" height="1615" alt="Optional Alt Text">

## Installation

You can install the WoE-IV-Bin Toolkit via pip:

``` bash

pip install woe_iv_bin==0.1.2
```

## Load Package
### Categorical WoE and IV Calculation
``` bash

from woe_iv_bin import categorical_woe
```

### Continuous WoE, IV Calculation and Binning Optimization
``` bash

from woe_iv_bin import continuous_woe
```

## Usage 
### Categorical WoE and IV Calculation

To calculate WoE and IV for categorical variables, use the `categorical_woe` function

``` bash
from woe_iv_bin import categorical_woe

woe_results = categorical_woe(df, 
                              cat_variable_name='cat_feature', 
                              y_df=df['target'])
print(woe_results)
```

### Continuous WoE, IV Calculation and Binning Optimization

For optimizing binning strategies for continuous features, use the `continuous_woe` function:

``` bash
from woe_iv_bin import continuous_woe

df_binned, optimized_bins, binning_woe_results = continuous_woe(df, 
                                                                feature = 'cont_feature', 
                                                                target = 'target', 
                                                                max_bins=20, 
                                                                min_samples_bin=0.05)
print(df_binned)
print(optimized_bins)
print(binning_woe_results)
```

## Ideal Uses
- **Credit Risk Assessment**: Evaluate the predictive power of categorical variables such as income brackets or loan types using WoE and IV analysis to inform credit risk assessment models.

- **Customer Segmentation**: Optimize binning strategies for continuous features like age or purchase amount to uncover meaningful customer segments with distinct behaviors and characteristics.

- **Marketing Effectiveness**: Assess the effectiveness of marketing campaigns by analyzing WoE and IV of categorical variables such as campaign channels or customer segments.

- **Insurance Underwriting**: Enhance risk assessment models by optimizing binning for continuous variables like property value or policy coverage amount.

## Contributing
Contributions to the WoE-IV-Bin Toolkit are highly appreciated! Whether it's bug fixes, feature enhancements, or documentation improvements, your contributions can help make the toolkit even more powerful and user-friendly for the community. Feel free to open issues, submit pull requests, or suggest new features on the project's GitHub repository.

## Documentation & Examples
For documentation and usage examples, visit the GitHub repository: https://github.com/knowusuboaky/woe-iv-bin

**Author**: Kwadwo Daddy Nyame Owusu - Boakye\
**Email**: kwadwo.owusuboakye@outlook.com\
**License**: MIT
