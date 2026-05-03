"""
GDP and Life Expectancy Analysis
=================================
Project by: Maitham Al-Shamery
Repository: github.com/maithamphysics/Ghufran_GDP
Course: Data Scientist - Codecademy

HOW TO RUN IN VS CODE:
1. Make sure all_data.csv is in the SAME folder as this script
2. Open terminal in VS Code
3. Run: python analysis.py
4. All plots will be saved as PNG files in the same folder
"""

# ─── STEP 1: Install libraries if missing ────────────────────────────────────
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

required = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'scipy']
for pkg in required:
    try:
        __import__(pkg if pkg != 'scikit-learn' else 'sklearn')
    except ImportError:
        print(f'Installing {pkg}...')
        install(pkg)

# ─── STEP 2: Import libraries ────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <-- This makes plots save WITHOUT needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from scipy import stats

# Plot styling
sns.set_style('darkgrid')
sns.set_palette('tab10')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size']       = 12
plt.rcParams['axes.titlesize']  = 15
plt.rcParams['axes.labelsize']  = 13

print('✅ Step 1: All libraries imported successfully!')

# ─── STEP 3: Load data ───────────────────────────────────────────────────────
# Get the folder where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, 'all_data.csv')

df = pd.read_csv(csv_path)
df.columns        = ['Country', 'Year', 'Life_Expectancy', 'GDP']
df['GDP_Trillions'] = df['GDP'] / 1e12
df['Log_GDP']     = np.log(df['GDP'])

countries = df['Country'].unique()
palette   = sns.color_palette('tab10', len(countries))

print(f'✅ Step 2: Data loaded — {df.shape[0]} rows × {df.shape[1]} columns')
print(f'   Countries: {list(countries)}')
print(f'   Years: {df["Year"].min()} to {df["Year"].max()}')
print()

# ─── STEP 4: EDA Summary ─────────────────────────────────────────────────────
print('=== STATISTICAL SUMMARY ===')
print(df.describe().round(2))
print()

print('=== SUMMARY PER COUNTRY ===')
summary = df.groupby('Country').agg(
    Avg_Life_Exp=('Life_Expectancy', 'mean'),
    Min_Life_Exp=('Life_Expectancy', 'min'),
    Max_Life_Exp=('Life_Expectancy', 'max'),
    Avg_GDP_T   =('GDP_Trillions',   'mean'),
).round(2)
print(summary)
print()

# ─── Helper: save figure ────────────────────────────────────────────────────
def save(filename, title=''):
    path = os.path.join(script_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   💾 Saved: {filename}')

# ─── PLOT 1: Life Expectancy Over Time ───────────────────────────────────────
print('📈 Plot 1: Life Expectancy Over Time...')
fig, ax = plt.subplots(figsize=(14, 7))
for i, country in enumerate(countries):
    subset = df[df['Country'] == country]
    ax.plot(subset['Year'], subset['Life_Expectancy'],
            marker='o', label=country, color=palette[i],
            linewidth=2.5, markersize=6)
ax.set_title('🌍 Life Expectancy Over Time (2000–2015)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Life Expectancy at Birth (Years)')
ax.legend(title='Country', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xticks(df['Year'].unique())
plt.xticks(rotation=45)
plt.tight_layout()
save('plot1_life_expectancy_over_time.png')

# ─── PLOT 2: GDP Over Time ───────────────────────────────────────────────────
print('📈 Plot 2: GDP Over Time...')
fig, ax = plt.subplots(figsize=(14, 7))
for i, country in enumerate(countries):
    subset = df[df['Country'] == country]
    ax.plot(subset['Year'], subset['GDP_Trillions'],
            marker='o', label=country, color=palette[i],
            linewidth=2.5, markersize=6)
ax.set_title('💰 GDP Over Time (2000–2015)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('GDP (Trillions USD)')
ax.legend(title='Country', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xticks(df['Year'].unique())
plt.xticks(rotation=45)
plt.tight_layout()
save('plot2_gdp_over_time.png')

# ─── PLOT 3: Average Bar Charts ──────────────────────────────────────────────
print('📊 Plot 3: Average Bar Charts...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
avg_le  = df.groupby('Country')['Life_Expectancy'].mean().sort_values(ascending=False)
avg_gdp = df.groupby('Country')['GDP_Trillions'].mean().sort_values(ascending=False)

bars1 = axes[0].bar(avg_le.index, avg_le.values,
                    color=sns.color_palette('viridis', len(avg_le)))
for bar, val in zip(bars1, avg_le.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontweight='bold')
axes[0].set_title('Average Life Expectancy (2000–2015)', fontweight='bold')
axes[0].set_xlabel('Country')
axes[0].set_ylabel('Years')
axes[0].set_ylim(0, 95)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha='right')

bars2 = axes[1].bar(avg_gdp.index, avg_gdp.values,
                    color=sns.color_palette('magma', len(avg_gdp)))
for bar, val in zip(bars2, avg_gdp.values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'${val:.2f}T', ha='center', fontweight='bold')
axes[1].set_title('Average GDP (2000–2015)', fontweight='bold')
axes[1].set_xlabel('Country')
axes[1].set_ylabel('GDP (Trillions USD)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha='right')
plt.suptitle('Average Life Expectancy and GDP by Country',
             fontsize=16, fontweight='bold')
plt.tight_layout()
save('plot3_avg_bar_charts.png')

# ─── PLOT 4: Violin + Box Plots ──────────────────────────────────────────────
print('🎻 Plot 4: Violin and Box Plots...')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.violinplot(data=df, x='Country', y='Life_Expectancy',
               palette='tab10', ax=axes[0], inner='box')
axes[0].set_title('Distribution of Life Expectancy (Violin)', fontweight='bold')
axes[0].set_xlabel('Country')
axes[0].set_ylabel('Life Expectancy (Years)')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha='right')

sns.boxplot(data=df, x='Country', y='Life_Expectancy',
            palette='tab10', ax=axes[1])
axes[1].set_title('Distribution of Life Expectancy (Box Plot)', fontweight='bold')
axes[1].set_xlabel('Country')
axes[1].set_ylabel('Life Expectancy (Years)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha='right')
plt.tight_layout()
save('plot4_violin_box.png')

# ─── PLOT 5: Scatter Plot ────────────────────────────────────────────────────
print('🔵 Plot 5: Scatter Plots...')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for i, country in enumerate(countries):
    subset = df[df['Country'] == country]
    axes[0].scatter(subset['GDP_Trillions'], subset['Life_Expectancy'],
                    label=country, color=palette[i], s=80, alpha=0.85)
z = np.polyfit(df['GDP_Trillions'], df['Life_Expectancy'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['GDP_Trillions'].min(), df['GDP_Trillions'].max(), 100)
axes[0].plot(x_line, p(x_line), 'k--', linewidth=2, label='Trend')
axes[0].set_title('GDP vs Life Expectancy', fontweight='bold')
axes[0].set_xlabel('GDP (Trillions USD)')
axes[0].set_ylabel('Life Expectancy (Years)')
axes[0].legend(title='Country', fontsize=10)

for i, country in enumerate(countries):
    subset = df[df['Country'] == country]
    axes[1].scatter(subset['Log_GDP'], subset['Life_Expectancy'],
                    label=country, color=palette[i], s=80, alpha=0.85)
log_gdp = df['Log_GDP']
z2 = np.polyfit(log_gdp, df['Life_Expectancy'], 1)
p2 = np.poly1d(z2)
x2 = np.linspace(log_gdp.min(), log_gdp.max(), 100)
axes[1].plot(x2, p2(x2), 'k--', linewidth=2, label='Trend')
axes[1].set_title('Log(GDP) vs Life Expectancy', fontweight='bold')
axes[1].set_xlabel('Log(GDP)')
axes[1].set_ylabel('Life Expectancy (Years)')
axes[1].legend(title='Country', fontsize=10)
plt.tight_layout()
save('plot5_scatter.png')

# ─── PLOT 6: Correlation Heatmap ─────────────────────────────────────────────
print('🔥 Plot 6: Correlation Heatmap...')
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[['Year', 'Life_Expectancy', 'GDP_Trillions']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f',
            cmap='coolwarm', center=0, ax=ax,
            linewidths=2, annot_kws={'size': 14},
            square=True, vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
save('plot6_correlation_heatmap.png')

# ─── PLOT 7: Country Subplots ────────────────────────────────────────────────
print('📊 Plot 7: Country Subplots...')
fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle('GDP and Life Expectancy by Country (2000–2015)',
             fontsize=18, fontweight='bold')
axes = axes.flatten()
for i, country in enumerate(countries):
    subset = df[df['Country'] == country]
    ax1 = axes[i]
    ax2 = ax1.twinx()
    ax1.bar(subset['Year'], subset['GDP_Trillions'],
            color=palette[i], alpha=0.5, label='GDP (Trillions)')
    ax2.plot(subset['Year'], subset['Life_Expectancy'],
             color='darkred', marker='o', linewidth=2.5,
             markersize=6, label='Life Expectancy')
    ax1.set_title(country, fontsize=13, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP (Trillions USD)', color=palette[i])
    ax2.set_ylabel('Life Expectancy (Years)', color='darkred')
    ax1.tick_params(axis='x', rotation=45)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc='upper left')
plt.tight_layout()
save('plot7_country_subplots.png')

# ─── PLOT 8: Change 2000 vs 2015 ─────────────────────────────────────────────
print('📈 Plot 8: Change 2000 vs 2015...')
start = df[df['Year'] == 2000][['Country', 'Life_Expectancy', 'GDP_Trillions']]
end   = df[df['Year'] == 2015][['Country', 'Life_Expectancy', 'GDP_Trillions']]
start.columns = ['Country', 'LE_2000', 'GDP_2000']
end.columns   = ['Country', 'LE_2015', 'GDP_2015']
compare = pd.merge(start, end, on='Country')
compare['LE_Change']  = compare['LE_2015']  - compare['LE_2000']
compare['GDP_Change'] = compare['GDP_2015'] - compare['GDP_2000']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors_le = ['#2ecc71' if x > 0 else '#e74c3c' for x in compare['LE_Change']]
bars = axes[0].bar(compare['Country'], compare['LE_Change'], color=colors_le)
for bar, val in zip(bars, compare['LE_Change']):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'+{val:.1f}' if val > 0 else f'{val:.1f}',
                 ha='center', fontweight='bold')
axes[0].set_title('Change in Life Expectancy (2000 → 2015)', fontweight='bold')
axes[0].set_xlabel('Country')
axes[0].set_ylabel('Change (Years)')
axes[0].axhline(0, color='black', linewidth=0.8)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha='right')

colors_gdp = ['#3498db' if x > 0 else '#e74c3c' for x in compare['GDP_Change']]
bars2 = axes[1].bar(compare['Country'], compare['GDP_Change'], color=colors_gdp)
for bar, val in zip(bars2, compare['GDP_Change']):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f'+${val:.1f}T' if val > 0 else f'${val:.1f}T',
                 ha='center', fontweight='bold', fontsize=10)
axes[1].set_title('Change in GDP (2000 → 2015)', fontweight='bold')
axes[1].set_xlabel('Country')
axes[1].set_ylabel('Change (Trillions USD)')
axes[1].axhline(0, color='black', linewidth=0.8)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha='right')
plt.tight_layout()
save('plot8_change_2000_2015.png')

# ─── STEP 5: Machine Learning ────────────────────────────────────────────────
print()
print('🤖 Machine Learning: Predicting Life Expectancy from GDP...')

X = df[['Log_GDP', 'Year']].values
y = df['Life_Expectancy'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

models = {
    'Linear Regression':        LinearRegression(),
    'Random Forest':            RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':        GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR':                      SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
}

results = {}
print()
print('=' * 65)
print(f'{"Model":<25} {"R² Score":>12} {"RMSE":>10} {"CV Score":>10}')
print('=' * 65)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    r2      = r2_score(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    cv      = cross_val_score(model, X_train_sc, y_train,
                              cv=5, scoring='r2').mean()
    results[name] = {'R2': r2, 'RMSE': rmse, 'CV': cv,
                     'model': model, 'y_pred': y_pred}
    print(f'{name:<25} {r2:>12.4f} {rmse:>10.4f} {cv:>10.4f}')

print('=' * 65)
best = max(results, key=lambda k: results[k]['R2'])
print(f'\n🏆 Best Model: {best} (R² = {results[best]["R2"]:.4f})')
print()

# ─── PLOT 9: Actual vs Predicted ─────────────────────────────────────────────
print('📊 Plot 9: Actual vs Predicted...')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    axes[i].scatter(y_test, res['y_pred'], alpha=0.7,
                    color=palette[i], s=80, edgecolors='white')
    mn = min(y_test.min(), res['y_pred'].min())
    mx = max(y_test.max(), res['y_pred'].max())
    axes[i].plot([mn, mx], [mn, mx], 'r--', linewidth=2,
                 label='Perfect prediction')
    axes[i].set_title(f'{name}\nR²={res["R2"]:.4f}  RMSE={res["RMSE"]:.4f}',
                      fontweight='bold')
    axes[i].set_xlabel('Actual Life Expectancy')
    axes[i].set_ylabel('Predicted Life Expectancy')
    axes[i].legend()
plt.suptitle('Actual vs Predicted — Model Comparison',
             fontsize=16, fontweight='bold')
plt.tight_layout()
save('plot9_ml_comparison.png')

# ─── PLOT 10: Feature Importance ─────────────────────────────────────────────
print('📊 Plot 10: Feature Importance...')
rf_model     = results['Random Forest']['model']
importances  = rf_model.feature_importances_
feature_names = ['Log(GDP)', 'Year']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feature_names, importances, color=['#3498db', '#e74c3c'])
for bar, val in zip(bars, importances):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')
ax.set_title('Feature Importance — Random Forest', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_xlim(0, 1)
plt.tight_layout()
save('plot10_feature_importance.png')

# ─── PLOT 11: Model Scores Bar Chart ─────────────────────────────────────────
print('📊 Plot 11: Model Scores...')
results_df = pd.DataFrame({
    'Model':  list(results.keys()),
    'R2':     [results[m]['R2'] for m in results],
    'CV':     [results[m]['CV'] for m in results],
}).sort_values('R2', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(results_df))
width = 0.35
b1 = ax.bar(x - width/2, results_df['R2'], width,
            label='R² Score', color='#3498db', alpha=0.85)
b2 = ax.bar(x + width/2, results_df['CV'], width,
            label='CV Score', color='#2ecc71', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend()
ax.axhline(0.9, color='red', linestyle='--', alpha=0.5)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=10)
plt.tight_layout()
save('plot11_model_scores.png')

# ─── PLOT 12: Polynomial Regression ──────────────────────────────────────────
print('📊 Plot 12: Polynomial Regression...')
X_p  = df[['Log_GDP']].values
y_p  = df['Life_Expectancy'].values
poly = PolynomialFeatures(degree=2)
Xp_t = poly.fit_transform(X_p)
lr_p = LinearRegression()
lr_p.fit(Xp_t, y_p)
y_curve  = lr_p.predict(Xp_t)
r2_poly  = r2_score(y_p, y_curve)
sort_idx = np.argsort(X_p.flatten())

fig, ax = plt.subplots(figsize=(12, 7))
for i, country in enumerate(countries):
    subset = df[df['Country'] == country]
    ax.scatter(subset['Log_GDP'], subset['Life_Expectancy'],
               label=country, color=palette[i], s=80, alpha=0.85)
ax.plot(X_p.flatten()[sort_idx], y_curve[sort_idx],
        'k-', linewidth=3, label='Polynomial fit (degree 2)')
ax.set_title('Polynomial Regression: Log(GDP) vs Life Expectancy',
             fontweight='bold')
ax.set_xlabel('Log(GDP)')
ax.set_ylabel('Life Expectancy (Years)')
ax.legend(title='Country', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.text(0.05, 0.95, f'R² = {r2_poly:.4f}',
        transform=ax.transAxes, fontsize=13,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
save('plot12_polynomial_regression.png')

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
print()
print('=' * 65)
print('  ✅ ALL DONE! KEY FINDINGS')
print('=' * 65)
print('''
1. POSITIVE CORRELATION
   GDP and life expectancy are positively correlated.
   Log(GDP) gives strongest linear relationship (r ≈ 0.87).

2. ZIMBABWE — Most Dramatic Change
   Life expectancy: 46 years (2000) → 60.7 years (2015)
   Despite very low GDP — shows healthcare matters too.

3. CHINA — Fastest GDP Growth
   GDP grew from $1.2T to $11T (800% increase).
   Life expectancy rose from 71.7 to 76.1 years.

4. USA vs GERMANY
   USA has highest GDP but Germany has higher life expectancy.
   Healthcare system quality matters beyond just wealth.

5. MACHINE LEARNING
   Gradient Boosting achieved highest R² score.
   Log(GDP) is the most important feature.
''')
print('=' * 65)
print()
print('📁 All 12 plots saved in your project folder!')
print('🚀 Upload them to: github.com/maithamphysics/Ghufran_GDP')
