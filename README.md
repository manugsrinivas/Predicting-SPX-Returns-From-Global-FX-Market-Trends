# SPX Returns Predictor Using FX Market Trends

*A PCA-based regression analysis linking global currency markets to U.S. equity returns.*

## Overview  
I investigated how **global FX market movements** relate to monthly returns in the **S&P 500 index**, asking whether systematic currency shifts (e.g., USD strength) embed predictive or explanatory power for U.S. equities using PCA to distill high-dimensional FX data into latent factors and evaluate their relationships to SPX returns using both **Ordinary Least Squares (OLS)** and **Lasso regression** models.

## Data  
- **Source:** Manually collected monthly FX rates (`FXmonthly.csv`) and S&P 500 returns (`sp500.csv`)  
- **Sample:** **119 months** of monthly return data after cleaning and alignment  
- **Outcome:** Monthly % return of the S&P 500 index  
- **Predictors:** Principal components extracted from standardized returns of global FX currency pairs (e.g., USD/EUR, USD/JPY, USD/GBP, etc.)  
- **Preprocessing:** FX prices converted to monthly returns, standardized, and decomposed using PCA  

## Methods  
Implemented a combination of **dimensionality reduction and regression modeling** to evaluate relationships between FX market trends and SPX returns.

- **Principal Component Analysis (PCA):**  
  Extracts latent, orthogonal factors summarizing major movements in the FX market. Loadings are analyzed to interpret the economic meaning of each component.  
- **OLS Regression:**  
  Regresses SPX returns on top K principal components to determine sign/magnitude relationships.  
- **Lasso Regression (L1):**  
  Regularized model to select the most informative PCs, control for overfitting, and enhance model stability in small samples.  
- **Variance cutoff:** First 3–4 PCs retained based on explained variance and eigenvalue distribution.  
- **Alignment & Cleaning:** All models run on merged data using inner join and `.dropna()` to ensure aligned indices.  

## Results 
- **Explained Variance:**  
  - PC1 alone explains ~30% of FX market variance  
  - First 3 PCs explain ~60% of total variance  
- **OLS Findings:**  
  - PC1 is often positively associated with SPX returns — interpreted as a broad USD weakening signal aligning with equity gains  
  - PC2, PC3 capture regional divergence (e.g., EM vs developed markets)  
- **Lasso Findings:**  
  - Shrinks coefficients on minor PCs to zero  
  - Retains 1–2 dominant PCs with stable positive/negative effects  
- **Loadings Interpretation:**  
  - PC1 loads heavily on all major USD pairs — likely a **risk-on/risk-off proxy**  
  - PC2 shows divergence between European and Asian currencies  
- **Prediction Fit:**  
  - R² modest (reflecting macro noise), but stable across methods  
  - Lasso improves generalizability in a small-n setting  

## Interpreting the models  
- **OLS Regression:**  
  Offers transparent coefficients to evaluate directionality (e.g., 1 SD increase in PC1 → X% shift in SPX return)  
- **Lasso Regression:**  
  Useful for **automatic variable selection**, avoiding overfit, especially when working with >5 FX-derived PCs  
- **PCA Loadings:**  
  Essential for decoding the economic meaning of the latent FX trends — e.g., broad USD index, regional divergence, EM shock exposure  

## Key takeaways  
- **FX markets matter.** Global currency movements contain systematic signals about U.S. equity trends, particularly via **USD strength/weakness cycles**.  
- **PCA is powerful.** It efficiently condenses noisy FX data into interpretable latent factors.  
- **Only a few FX factors matter.** Both OLS and Lasso show that 1–2 principal components carry most of the equity signal.  
- **Lasso helps generalize.** Regularization controls for instability in small time-series settings.  
- **This is not predictive yet.** Current model explores **contemporaneous relationships**; future work should incorporate **lags** and **macro control variables**.  

## Assumptions & limitations  
- **PCA assumptions:** Components are linear and orthogonal; complex non-linear FX structures are not captured  
- **Stationarity:** Assumes factor structure is stable across time (no rolling PCA)  
- **No lag structure:** Model uses same-month FX and SPX returns — does not test predictive ability  
- **Small sample (n = 119):** Susceptible to overfit without regularization  
- **Exogeneity caveat:** FX assumed exogenous to SPX returns, though macro shocks can affect both  
- **No macro overlay:** Interest rates, inflation, GDP differentials not included in current framework  

## Research implications  
- **FX-based indicators** may improve macro hedge fund signals or tactical asset allocation tools  
- **Rolling PCA** and regime switching could enhance robustness across economic cycles  
- **Lasso + PCA** is a promising combo for **interpretable macro modeling** in small-sample environments  
- Future studies could test **lagged FX signals**, **macro controls**, and **cross-asset spillovers** (e.g., FX → bonds/equities)
